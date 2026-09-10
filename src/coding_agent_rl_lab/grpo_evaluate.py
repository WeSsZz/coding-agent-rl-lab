"""Evaluate frozen adapters with the same TRL tool loop used by GRPO."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .grpo_remote import RemoteGRPOCodingEnvironment
from .grpo_train import (
    configure_prompt_rows_tool_format, configure_tool_response_parsing,
    load_prompt_rows, read_worker_token, probe_bare_json_tool_parsing,
)
from .swe_gym_smoke import pinned_rows_for_task_set


def select_rows(rows: list[dict[str, Any]], task_set: str) -> list[dict[str, Any]]:
    if task_set not in {"train", "regression"}:
        raise ValueError("direction checks accept only train or regression")
    by_id = {row["task_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate prompt task IDs")
    return [by_id[item.instance_id] for item in pinned_rows_for_task_set(task_set)]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records or any(not isinstance(r.get("reward_components"), dict) for r in records):
        raise ValueError("complete verifier reward components are required")
    components = [r["reward_components"] for r in records]
    return {
        "trial_count": len(records),
        "strict_successes": sum(bool(c["strict_success"]) for c in components),
        "mean_shaped_reward": sum(r["reward"] for r in records) / len(records),
        "patches_created": sum(bool(c["patch_created"]) for c in components),
        "valid_patches": sum(bool(c["patch_valid"]) for c in components),
        "verified_patches": sum(bool(c["patch_created"] and c["verifier_run_after_patch"]) for c in components),
        "trials_resolving_failures": sum(
            c["patch_valid"] and not c["violations"]
            and c["baseline_failure_count"] is not None
            and c["final_failure_count"] is not None
            and c["final_failure_count"] < c["baseline_failure_count"]
            for c in components
        ),
        "trials_with_unknown_final_failure_count": sum(c["final_failure_count"] is None for c in components),
        "trials_with_new_failures": sum((c["new_failure_count"] or 0) > 0 for c in components),
        "trials_with_violations": sum(bool(c["violations"]) for c in components),
    }


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_resume(previous: dict[str, Any], expected: dict[str, Any], task_ids: set[str]) -> None:
    for key in ("adapter_path", "adapter_sha256", "model_path", "prompt_rows_sha256", "seed", "budget", "planned_trial_count"):
        if previous.get(key) != expected[key]:
            raise ValueError(f"incompatible resume field: {key}")
    saved_ids = [entry["task_id"] for entry in previous["tasks"]]
    if len(saved_ids) != len(set(saved_ids)) or not set(saved_ids) <= task_ids:
        raise ValueError("resume contains duplicate or unexpected tasks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--prompt-rows", required=True)
    parser.add_argument("--worker-token-file", required=True)
    parser.add_argument("--worker-base-url", default="http://127.0.0.1:9011")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=81000)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.num_generations < 2:
        parser.error("--num-generations must be at least two")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=args.resume)
    rows = load_prompt_rows(Path(args.prompt_rows))
    selected = [(split, row) for split in ("train", "regression") for row in select_rows(rows, split)]
    token = read_worker_token(Path(args.worker_token_file))
    adapter = Path(args.adapter_path).resolve()
    before = file_hash(adapter / "adapter_model.safetensors")
    report: dict[str, Any] = {
        "schema_version": 1, "training_performed": False, "run_complete": False,
        "adapter_path": str(adapter), "adapter_sha256": before,
        "model_path": args.model_path, "prompt_rows_sha256": file_hash(Path(args.prompt_rows)),
        "seed": args.seed, "planned_trial_count": len(selected) * args.num_generations,
        "protocol": "TRL GRPO evaluate, bare JSON, dynamic tools including replace_lines",
        "budget": {"max_completion_length": 4096, "max_tool_calling_iterations": 8,
                   "temperature": 1.0, "top_p": 0.95, "num_generations": args.num_generations},
        "tasks": [],
    }
    if args.resume and (output / "report.json").exists():
        previous = json.loads((output / "report.json").read_text())
        validate_resume(previous, report, {row["task_id"] for _, row in selected})
        report["tasks"] = previous["tasks"]

    def checkpoint() -> None:
        temporary = output / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "report.json")

    checkpoint()
    import torch
    import transformers
    import trl
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    report["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__}
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    configure_tool_response_parsing(tokenizer, bare_json_tool_calls=True)
    if not probe_bare_json_tool_parsing(tokenizer, lambda t, ids, *, prefix: t.parse_response(ids, prefix=prefix)):
        raise RuntimeError("bare JSON parser probe failed")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16, local_files_only=True)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.requires_grad_(False)
    model.eval()
    environments: list[RemoteGRPOCodingEnvironment] = []
    audit_path = output / "unused.jsonl"

    def factory() -> RemoteGRPOCodingEnvironment:
        environment = RemoteGRPOCodingEnvironment(args.worker_base_url, token, reward_audit_path=audit_path)
        environments.append(environment)
        return environment

    config = GRPOConfig(
        output_dir=str(output / "trainer"), max_steps=1,
        per_device_train_batch_size=1, per_device_eval_batch_size=args.num_generations,
        generation_batch_size=args.num_generations, num_generations=args.num_generations,
        num_generations_eval=args.num_generations, max_completion_length=4096,
        max_tool_calling_iterations=8, temperature=1.0, top_p=0.95,
        bf16=True, use_vllm=False, report_to="none", save_strategy="no", seed=args.seed,
    )
    trainer = GRPOTrainer(
        model=model, reward_funcs=None, args=config, processing_class=tokenizer,
        train_dataset=Dataset.from_list(configure_prompt_rows_tool_format([selected[0][1]], bare_json_tool_calls=True)),
        environment_factory=factory,
    )
    all_records: dict[str, list[dict[str, Any]]] = {"train": [], "regression": []}
    completed = {entry["task_id"]: entry for entry in report["tasks"]}
    try:
        for index, (split, row) in enumerate(selected):
            task_id = row["task_id"]
            audit_path = output / f"{task_id}-reward-audit.jsonl"
            if task_id in completed:
                records = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
                if len(records) != args.num_generations or any(r["task_id"] != task_id for r in records):
                    raise ValueError("completed task has incomplete or mismatched audit")
                entry = completed[task_id]
                if entry["split"] != split or entry["seed"] != args.seed + index * 100:
                    raise ValueError("completed task split or seed mismatch")
                entry["summary"] = summarize(records)
                all_records[split].extend(records)
                continue
            if audit_path.exists():
                audit_path.rename(audit_path.with_suffix(f".interrupted-{time.time_ns()}.jsonl"))
            for environment in environments:
                environment._reward_audit_path = audit_path
            seed = args.seed + index * 100
            set_seed(seed)
            started = time.monotonic()
            dataset = Dataset.from_list(configure_prompt_rows_tool_format([row], bare_json_tool_calls=True))
            metrics = trainer.evaluate(eval_dataset=dataset)
            records = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
            if len(records) != args.num_generations or any(r["task_id"] != task_id for r in records):
                raise RuntimeError("incomplete or mismatched reward audit; refusing a capability summary")
            entry = {"task_id": task_id, "split": split, "seed": seed,
                     "elapsed_seconds": time.monotonic() - started,
                     "summary": summarize(records), "metrics": metrics}
            report["tasks"].append(entry)
            all_records[split].extend(records)
            report["completed_trial_count"] = sum(len(v) for v in all_records.values())
            checkpoint()
            print(json.dumps(entry), flush=True)
        report["summary"] = {split: summarize(records) for split, records in all_records.items()}
        report["adapter_unchanged"] = before == file_hash(adapter / "adapter_model.safetensors")
        report["optimizer_steps"] = trainer.state.global_step
        if not report["adapter_unchanged"] or report["optimizer_steps"] != 0:
            raise RuntimeError("evaluation mutated the adapter or performed optimizer steps")
        report["run_complete"] = True
        checkpoint()
    finally:
        for environment in environments:
            environment._delete()


if __name__ == "__main__":
    main()
