from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .model_policy import PROMPT_VERSION, OpenAICompatiblePolicy
from .sft_grpo import GRPO_ACTION_PROTOCOL, GRPO_SFT_PROMPT_VERSION, GRPO_SFT_SCHEMA
from .swe_gym_smoke import pinned_rows_for_task_set


class SFTTrainingError(RuntimeError):
    pass


def load_sft_examples(
    dataset_path: Path,
    report_path: Path,
    *,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = _load_json_object(report_path, "dataset report")
    _validate_dataset_report(report)
    allowed_task_ids = {item.instance_id for item in pinned_rows_for_task_set("train")}
    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    with dataset_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SFTTrainingError(f"invalid JSON on SFT row {line_number}") from exc
            _validate_sft_example(
                example,
                line_number,
                allowed_task_ids,
                dataset_schema=report["dataset_schema"],
                prompt_version=report["prompt_version"],
            )
            example_id = example["example_id"]
            if example_id in seen:
                raise SFTTrainingError(f"duplicate SFT example_id: {example_id}")
            seen.add(example_id)
            examples.append(example)
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise SFTTrainingError("SFT dataset must contain at least one example")
    if limit is None and report["example_count"] != len(examples):
        raise SFTTrainingError("dataset example count does not match its report")
    report_task_ids = set(report["task_ids"])
    example_task_ids = {example["task_id"] for example in examples}
    if not example_task_ids.issubset(report_task_ids):
        raise SFTTrainingError("SFT examples contain task ids absent from the dataset report")
    return examples, report


def prepare_prompt_completion_rows(
    examples: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "prompt": example["messages"][:-1],
            "completion": [example["messages"][-1]],
        }
        for example in examples
    ]


def build_token_length_report(
    tokenizer: Any,
    rows: Iterable[dict[str, Any]],
    *,
    max_length: int,
) -> dict[str, int]:
    if max_length <= 0:
        raise SFTTrainingError("max_length must be positive")
    full_lengths: list[int] = []
    completion_lengths: list[int] = []
    for row in rows:
        prompt = row["prompt"]
        completion = row["completion"]
        rendered = tokenizer.apply_chat_template(
            [*prompt, *completion],
            tokenize=False,
            add_generation_prompt=False,
        )
        if not isinstance(rendered, str):
            raise SFTTrainingError("chat template must render text before token length inspection")
        full_ids = tokenizer.encode(rendered, add_special_tokens=False)
        completion_ids = tokenizer.encode(
            completion[0]["content"],
            add_special_tokens=False,
        )
        full_lengths.append(len(full_ids))
        completion_lengths.append(len(completion_ids))
    over_limit = sum(length > max_length for length in full_lengths)
    return {
        "example_count": len(full_lengths),
        "min_full_tokens": min(full_lengths),
        "max_full_tokens": max(full_lengths),
        "max_completion_tokens": max(completion_lengths),
        "over_max_length_count": over_limit,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight or train a train-only answer-supervised LoRA tool warm-start"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-report", required=True)
    parser.add_argument("--output-dir", default="/root/autodl-tmp/sft-warm-start")
    parser.add_argument("--example-limit", type=int)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=61001)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Perform LoRA updates. Without this flag, run validation and tokenizer preflight only.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    model_path = Path(args.model_path).resolve()
    if not model_path.is_dir():
        raise SystemExit(f"model path is not a directory: {model_path}")
    dataset_path = Path(args.dataset).resolve()
    report_path = Path(args.dataset_report).resolve()
    examples, dataset_report = load_sft_examples(
        dataset_path,
        report_path,
        limit=args.example_limit,
    )
    training_rows = prepare_prompt_completion_rows(examples)

    try:
        from datasets import Dataset
        import torch
        import transformers
        import trl
        from peft import LoraConfig
        from transformers import AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(f"SFT dependencies are unavailable: {exc}") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    token_lengths = build_token_length_report(
        tokenizer,
        training_rows,
        max_length=args.max_length,
    )
    if token_lengths["over_max_length_count"]:
        raise SystemExit(
            f"{token_lengths['over_max_length_count']} SFT examples exceed --max-length; "
            "refuse to truncate answer-supervised actions"
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "dataset_schema": dataset_report["dataset_schema"],
        "dataset_task_set": dataset_report["task_set"],
        "dataset_contains_answers": dataset_report["contains_answers"],
        "prompt_version": dataset_report["prompt_version"],
        "action_protocol": dataset_report.get("action_protocol", "model-policy-json"),
        "example_count": len(examples),
        "token_lengths": token_lengths,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "trl_version": trl.__version__,
        "cuda_available": torch.cuda.is_available(),
        "completion_only_loss": True,
        "training_performed": False,
    }
    if not args.train:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for SFT training")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        use_cache=False,
        model_init_kwargs={
            "dtype": "bfloat16",
            "local_files_only": True,
            "trust_remote_code": False,
        },
        max_length=args.max_length,
        truncation_mode="keep_start",
        completion_only_loss=True,
        packing=False,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    trainer = SFTTrainer(
        model=str(model_path),
        args=training_args,
        train_dataset=Dataset.from_list(training_rows),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    train_result = trainer.train()
    adapter_path = output_dir / "final-adapter"
    trainer.save_model(str(adapter_path))
    final_metrics = next(
        (
            entry
            for entry in reversed(trainer.state.log_history)
            if "loss" in entry or "grad_norm" in entry
        ),
        {},
    )
    grad_norm = _metric_number(final_metrics.get("grad_norm"))
    report["training_performed"] = True
    report["optimizer_steps"] = trainer.state.global_step
    report["effective_update"] = bool(grad_norm is not None and grad_norm > 0.0)
    report["training_metrics"] = {
        "train_loss": _metric_number(train_result.metrics.get("train_loss")),
        "loss": _metric_number(final_metrics.get("loss")),
        "grad_norm": grad_norm,
    }
    report["output_dir"] = str(output_dir)
    report["adapter_path"] = str(adapter_path)
    (output_dir / "training-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _validate_dataset_report(report: dict[str, Any]) -> None:
    schema = report.get("dataset_schema")
    if schema not in {"coding-agent-gold-sft-v1", GRPO_SFT_SCHEMA}:
        raise SFTTrainingError("unsupported SFT dataset schema")
    if report.get("task_set") != "train":
        raise SFTTrainingError("SFT dataset report must be train-only")
    if report.get("contains_answers") is not True:
        raise SFTTrainingError("SFT dataset report must explicitly declare contains_answers=true")
    if report.get("answer_source") != "official_swe_gym_gold_patch":
        raise SFTTrainingError("SFT dataset report has an unsupported answer source")
    expected_prompt = (
        GRPO_SFT_PROMPT_VERSION if schema == GRPO_SFT_SCHEMA else PROMPT_VERSION
    )
    if report.get("prompt_version") != expected_prompt:
        raise SFTTrainingError("SFT dataset prompt version does not match the current policy")
    if schema == GRPO_SFT_SCHEMA and report.get("action_protocol") != GRPO_ACTION_PROTOCOL:
        raise SFTTrainingError("GRPO SFT dataset has an unsupported action protocol")
    task_ids = report.get("task_ids")
    if not isinstance(task_ids, list) or not task_ids:
        raise SFTTrainingError("SFT dataset report must list task ids")
    allowed = {item.instance_id for item in pinned_rows_for_task_set("train")}
    if any(not isinstance(task_id, str) or task_id not in allowed for task_id in task_ids):
        raise SFTTrainingError("SFT dataset report contains a non-train task")
    if not isinstance(report.get("example_count"), int) or report["example_count"] <= 0:
        raise SFTTrainingError("SFT dataset report must declare a positive example count")


def _validate_sft_example(
    example: Any,
    line_number: int,
    allowed_task_ids: set[str],
    *,
    dataset_schema: str,
    prompt_version: str,
) -> None:
    if not isinstance(example, dict) or example.get("schema_version") != 1:
        raise SFTTrainingError(f"invalid SFT schema on row {line_number}")
    if example.get("task_set") != "train" or example.get("task_id") not in allowed_task_ids:
        raise SFTTrainingError(f"SFT row {line_number} is outside the train split")
    if example.get("contains_answers") is not True:
        raise SFTTrainingError(f"SFT row {line_number} does not declare contains_answers=true")
    if example.get("answer_source") != "official_swe_gym_gold_patch":
        raise SFTTrainingError(f"SFT row {line_number} has an unsupported answer source")
    if example.get("prompt_version") != prompt_version:
        raise SFTTrainingError(f"SFT row {line_number} has a stale prompt version")
    example_id = example.get("example_id")
    if not isinstance(example_id, str) or not example_id:
        raise SFTTrainingError(f"SFT row {line_number} has no example_id")
    messages = example.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 3
        or [message.get("role") for message in messages if isinstance(message, dict)]
        != ["system", "user", "assistant"]
        or any(not isinstance(message.get("content"), str) for message in messages)
    ):
        raise SFTTrainingError(f"SFT row {line_number} has invalid conversational messages")
    target_action = example.get("target_action")
    if not isinstance(target_action, dict):
        raise SFTTrainingError(f"SFT row {line_number} has no target action")
    if dataset_schema == GRPO_SFT_SCHEMA:
        if example.get("action_protocol") != GRPO_ACTION_PROTOCOL:
            raise SFTTrainingError(f"SFT row {line_number} has an unsupported action protocol")
        try:
            tool_call = json.loads(messages[-1]["content"])
        except json.JSONDecodeError as exc:
            raise SFTTrainingError(f"SFT row {line_number} has an invalid assistant action") from exc
        expected = {
            "name": target_action.get("kind"),
            "arguments": target_action.get("arguments", {}),
        }
        if tool_call != expected or example.get("target_tool_call") != expected:
            raise SFTTrainingError(
                f"SFT row {line_number} assistant content disagrees with target action"
            )
    else:
        try:
            parsed = OpenAICompatiblePolicy._parse_action(messages[-1]["content"])
        except ValueError as exc:
            raise SFTTrainingError(f"SFT row {line_number} has an invalid assistant action") from exc
        if parsed.to_dict() != target_action:
            raise SFTTrainingError(
                f"SFT row {line_number} assistant content disagrees with target action"
            )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SFTTrainingError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise SFTTrainingError(f"{label} must be a JSON object")
    return value


def _validate_args(args: argparse.Namespace) -> None:
    if args.example_limit is not None and args.example_limit <= 0:
        raise SystemExit("--example-limit must be positive")
    if args.max_length <= 0:
        raise SystemExit("--max-length must be positive")
    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise SystemExit("--gradient-accumulation-steps must be positive")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive")


def _metric_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


if __name__ == "__main__":
    main()
