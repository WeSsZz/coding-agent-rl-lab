from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from coding_agent_rl_lab.lora_inference import merge_lora_adapter_for_inference
from coding_agent_rl_lab.sft_train import (
    load_sft_examples,
    prepare_prompt_completion_rows,
)


class TargetScoreError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score verified semantic-recovery next actions without updating weights"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=16384)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise TargetScoreError(f"refusing to overwrite output: {output}")
    model_path = Path(args.model_path).resolve()
    adapter_path = Path(args.adapter_path).resolve()
    for label, path in (("model", model_path), ("adapter", adapter_path)):
        if not path.is_dir():
            raise TargetScoreError(f"{label} path is not a directory: {path}")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.data_utils import _tokenize

    dataset_path = Path(args.dataset)
    report_path = Path(args.dataset_report)
    examples, _ = load_sft_examples(dataset_path, report_path)
    rows = prepare_prompt_completion_rows(examples)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    if not torch.cuda.is_available():
        raise TargetScoreError("CUDA is required for target scoring")

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
    )
    base.to("cuda:0")
    merge_report = merge_lora_adapter_for_inference(base, adapter_path)
    model = base
    model.requires_grad_(False)
    model.eval()
    device = next(model.parameters()).device

    scored: list[dict[str, Any]] = []
    groups: dict[str, list[float]] = defaultdict(list)
    correct_by_group: dict[str, list[float]] = defaultdict(list)
    with torch.inference_mode():
        for example, row in zip(examples, rows, strict=True):
            prompt_ids = _tokenize(
                tokenizer,
                row["prompt"],
                add_generation_prompt=True,
                tools=row.get("tools"),
            )["input_ids"]
            full_ids = _tokenize(
                tokenizer,
                [*row["prompt"], *row["completion"]],
                tools=row.get("tools"),
            )["input_ids"]
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise TargetScoreError(f"prompt boundary mismatch: {example['example_id']}")
            completion_length = len(full_ids) - len(prompt_ids)
            if completion_length <= 0 or len(full_ids) > args.max_length:
                raise TargetScoreError(f"invalid target length: {example['example_id']}")
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            try:
                model_output = model(
                    input_ids=input_ids,
                    use_cache=False,
                    logits_to_keep=completion_length + 1,
                    return_dict=True,
                )
                logits = model_output.logits[0, :-1].float()
            except TypeError:
                model_output = model(input_ids=input_ids, use_cache=False, return_dict=True)
                logits = model_output.logits[0, len(prompt_ids) - 1 : -1].float()
            targets = input_ids[0, -completion_length:]
            if logits.shape[0] != targets.shape[0]:
                raise TargetScoreError(
                    f"logit/target length mismatch: {example['example_id']} "
                    f"{tuple(logits.shape)} vs {tuple(targets.shape)}"
                )
            losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
            predictions = logits.argmax(dim=-1)
            nll = float(losses.mean().item())
            accuracy = float((predictions == targets).float().mean().item())
            kind = str(example["target_action"]["kind"])
            group = _action_group(kind)
            groups[group].append(nll)
            correct_by_group[group].append(accuracy)
            recovery = example.get("semantic_recovery", {})
            scored.append(
                {
                    "example_id": example["example_id"],
                    "state_id": recovery.get("state_id"),
                    "sequence": recovery.get("sequence"),
                    "target_kind": kind,
                    "action_group": group,
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": completion_length,
                    "mean_nll": nll,
                    "perplexity": math.exp(min(nll, 20.0)),
                    "token_accuracy": accuracy,
                }
            )
            del input_ids, model_output, logits, targets, losses, predictions

    report = {
        "schema_version": 1,
        "completed": True,
        "training_performed": False,
        "model_path": str(model_path),
        "adapter_path": str(adapter_path),
        "model_config_sha256": _optional_sha256(model_path / "config.json"),
        "adapter_config_sha256": _optional_sha256(adapter_path / "adapter_config.json"),
        "adapter_weights_sha256": _optional_sha256(
            adapter_path / "adapter_model.safetensors"
        ),
        "dataset_sha256": _sha256(dataset_path),
        "dataset_report_sha256": _sha256(report_path),
        "example_count": len(scored),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "adapter_merge": merge_report,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "mean_nll": sum(item["mean_nll"] for item in scored) / len(scored),
        "mean_token_accuracy": sum(item["token_accuracy"] for item in scored) / len(scored),
        "by_action_group": {
            group: {
                "count": len(values),
                "mean_nll": sum(values) / len(values),
                "mean_token_accuracy": sum(correct_by_group[group])
                / len(correct_by_group[group]),
            }
            for group, values in sorted(groups.items())
        },
        "rows": scored,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "adapter_weights_sha256",
                    "example_count",
                    "elapsed_seconds",
                    "peak_cuda_memory_bytes",
                    "mean_nll",
                    "mean_token_accuracy",
                    "by_action_group",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _action_group(kind: str) -> str:
    if kind in {"search_text", "list_files", "read_file"}:
        return "navigate_or_read"
    if kind in {"replace_text", "replace_lines"}:
        return "edit_or_repair"
    if kind in {"run_tests", "finish"}:
        return "verify"
    return "other"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _optional_sha256(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


if __name__ == "__main__":
    main()
