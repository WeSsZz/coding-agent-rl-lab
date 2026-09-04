from __future__ import annotations

import argparse
import copy
import json
import os
import stat
from pathlib import Path
from typing import Any

from .grpo_remote import RemoteGRPOCodingEnvironment


class GRPOTrainingError(RuntimeError):
    pass


_BARE_JSON_TOOL_RESPONSE_TEMPLATE: dict[str, Any] = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {
        "tool_calls": {
            "close": "<|im_end|>",
            "content": "json",
            "transform": [{"type": "function", "function": "{content}"}],
        }
    },
}

_BARE_JSON_TOOL_CALL_INSTRUCTION = """Every assistant turn must contain exactly one bare JSON
object and no prose or tags. Use this exact shape:
{"name":"search_text","arguments":{"query":"literal identifier"}}
Replace the example name and arguments with the selected provided tool. Never use Markdown fences,
<tool_call> tags, a "kind" field, or natural-language explanation."""


def load_prompt_rows(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GRPOTrainingError(f"invalid JSON on prompt row {line_number}") from exc
            _validate_prompt_row(row, line_number)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise GRPOTrainingError("prompt row file must contain at least one task")
    return rows


def read_worker_token(path: Path) -> str:
    token = path.read_text(encoding="utf-8").strip()
    if len(token) < 32:
        raise GRPOTrainingError("worker token must contain at least 32 characters")
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise GRPOTrainingError("worker token file must not be accessible by group or others")
    return token


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight or run single-GPU LoRA GRPO against the remote Docker worker"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--adapter-path",
        help="Optional trainable SFT LoRA adapter used to initialize GRPO.",
    )
    parser.add_argument("--prompt-rows", required=True)
    parser.add_argument("--worker-token-file", required=True)
    parser.add_argument("--worker-base-url", default="http://127.0.0.1:9010")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/grpo-output")
    parser.add_argument("--task-count", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--max-tool-calling-iterations", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--bare-json-tool-calls",
        action="store_true",
        help=(
            "Parse a single bare JSON {name, arguments} object as a tool call. "
            "Use only for policies whose audited rollout protocol emits this format."
        ),
    )
    parser.add_argument(
        "--log-completions",
        action="store_true",
        help="Print sampled completions for an audited smoke diagnosis.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Perform weight updates. Without this flag only a read-only preflight runs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    model_path = Path(args.model_path).resolve()
    if not model_path.is_dir():
        raise SystemExit(f"model path is not a directory: {model_path}")
    adapter_path = Path(args.adapter_path).resolve() if args.adapter_path else None
    if adapter_path is not None and not adapter_path.is_dir():
        raise SystemExit(f"adapter path is not a directory: {adapter_path}")
    rows = configure_prompt_rows_tool_format(
        load_prompt_rows(Path(args.prompt_rows), limit=args.task_count),
        bare_json_tool_calls=args.bare_json_tool_calls,
    )
    token = read_worker_token(Path(args.worker_token_file))

    try:
        from datasets import Dataset
        import torch
        import transformers
        import trl
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
        from trl.chat_template_utils import supports_tool_calling
    except ImportError as exc:
        raise SystemExit(f"GRPO dependencies are unavailable: {exc}") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    configure_tool_response_parsing(
        tokenizer,
        bare_json_tool_calls=args.bare_json_tool_calls,
    )
    tool_calling_supported = bool(supports_tool_calling(tokenizer))
    if not tool_calling_supported:
        raise SystemExit("the model tokenizer/chat template does not support TRL tool calling")
    bare_json_parsing_supported: bool | None = None
    if args.bare_json_tool_calls:
        bare_json_probe = run_bare_json_tool_parsing_probe(
            tokenizer,
            lambda active_tokenizer, ids, *, prefix: active_tokenizer.parse_response(
                ids,
                prefix=prefix,
            ),
        )
        bare_json_parsing_supported = is_valid_bare_json_tool_probe(bare_json_probe)
        if not bare_json_parsing_supported:
            raise SystemExit(
                "bare JSON tool-call response parsing probe failed: "
                + json.dumps(bare_json_probe, ensure_ascii=False)
            )

    def environment_factory() -> RemoteGRPOCodingEnvironment:
        return RemoteGRPOCodingEnvironment(args.worker_base_url, token)

    worker_probe = environment_factory()
    try:
        initial_observation = worker_probe.reset(**rows[0])
        file_listing = worker_probe.list_files()
    finally:
        worker_probe._delete()

    report: dict[str, Any] = {
        "schema_version": 1,
        "model_path": str(model_path),
        "initial_adapter_path": str(adapter_path) if adapter_path is not None else None,
        "task_count": len(rows),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "trl_version": trl.__version__,
        "cuda_available": torch.cuda.is_available(),
        "tool_calling_supported": tool_calling_supported,
        "tool_response_format": (
            "bare_json" if args.bare_json_tool_calls else "model_default"
        ),
        "bare_json_tool_call_parsing_supported": bare_json_parsing_supported,
        "worker_baseline_received": "Baseline verifier result:" in initial_observation,
        "worker_baseline_failed": "Tests failed" in initial_observation,
        "worker_file_listing_received": bool(file_listing.strip()),
        "training_performed": False,
    }
    if not args.train:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for training")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        use_cache=False,
        model_init_kwargs=(
            None
            if adapter_path is not None
            else {
                "dtype": "bfloat16",
                "local_files_only": True,
                "trust_remote_code": False,
            }
        ),
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=args.max_tool_calling_iterations,
        use_vllm=False,
        temperature=1.0,
        top_p=0.95,
        logging_steps=1,
        logging_first_step=True,
        log_completions=args.log_completions,
        num_completions_to_print=args.num_generations if args.log_completions else None,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )
    trainer_model: Any = str(model_path)
    peft_config: LoraConfig | None = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    if adapter_path is not None:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype="bfloat16",
            local_files_only=True,
            trust_remote_code=False,
        )
        trainer_model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=True,
        )
        peft_config = None
    trainer = GRPOTrainer(
        model=trainer_model,
        reward_funcs=None,
        args=training_args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        peft_config=peft_config,
        environment_factory=environment_factory,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir / "final-adapter"))
    step_metrics = next(
        (
            entry
            for entry in reversed(trainer.state.log_history)
            if "grad_norm" in entry or "reward_std" in entry
        ),
        {},
    )
    grad_norm = _metric_number(step_metrics.get("grad_norm"))
    report["training_performed"] = True
    report["optimizer_steps"] = trainer.state.global_step
    report["effective_update"] = bool(grad_norm is not None and grad_norm > 0.0)
    report["training_metrics"] = {
        "train_loss": _metric_number(train_result.metrics.get("train_loss")),
        "grad_norm": grad_norm,
        "reward": _metric_number(step_metrics.get("reward")),
        "reward_std": _metric_number(step_metrics.get("reward_std")),
        "zero_reward_std_fraction": _metric_number(step_metrics.get("frac_reward_zero_std")),
        "tool_call_frequency": _metric_number(step_metrics.get("tools/call_frequency")),
        "tool_failure_frequency": _metric_number(step_metrics.get("tools/failure_frequency")),
    }
    report["output_dir"] = str(output_dir)
    (output_dir / "training-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _validate_prompt_row(row: Any, line_number: int) -> None:
    if not isinstance(row, dict):
        raise GRPOTrainingError(f"prompt row {line_number} must be an object")
    if not isinstance(row.get("task_id"), str) or not row["task_id"]:
        raise GRPOTrainingError(f"prompt row {line_number} requires task_id")
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        raise GRPOTrainingError(f"prompt row {line_number} requires a non-empty prompt")
    for message in prompt:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user"}:
            raise GRPOTrainingError(f"prompt row {line_number} contains an invalid message")
        if not isinstance(message.get("content"), str):
            raise GRPOTrainingError(f"prompt row {line_number} message content must be text")


def _validate_args(args: argparse.Namespace) -> None:
    if args.task_count <= 0:
        raise SystemExit("--task-count must be positive")
    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    if args.num_generations < 2:
        raise SystemExit("--num-generations must be at least 2 for group-relative rewards")
    if args.max_completion_length <= 0:
        raise SystemExit("--max-completion-length must be positive")
    if args.max_tool_calling_iterations <= 0:
        raise SystemExit("--max-tool-calling-iterations must be positive")


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


def configure_tool_response_parsing(
    tokenizer: Any,
    *,
    bare_json_tool_calls: bool,
) -> Any:
    """Select the response parser without changing the model chat template.

    Qwen2.5-Coder reliably emits this project's audited v8 action protocol as
    a bare ``{"name": ..., "arguments": ...}`` object.  TRL's default Qwen
    response template only recognizes the same object inside ``<tool_call>``
    tags, so an explicit compatibility mode is required for the environment
    loop to execute otherwise-valid actions.
    """

    if bare_json_tool_calls:
        tokenizer.response_template = copy.deepcopy(_BARE_JSON_TOOL_RESPONSE_TEMPLATE)
    return tokenizer


def configure_prompt_rows_tool_format(
    rows: list[dict[str, Any]],
    *,
    bare_json_tool_calls: bool,
) -> list[dict[str, Any]]:
    configured = copy.deepcopy(rows)
    if not bare_json_tool_calls:
        return configured
    marker = "\nEvery assistant turn"
    for row in configured:
        for message in row["prompt"]:
            if message["role"] != "system":
                continue
            prefix, separator, _ = message["content"].partition(marker)
            if not separator:
                prefix = message["content"].rstrip()
            message["content"] = prefix.rstrip() + "\n\n" + _BARE_JSON_TOOL_CALL_INSTRUCTION
    return configured


def run_bare_json_tool_parsing_probe(tokenizer: Any, parse_response_fn: Any) -> dict[str, Any]:
    """Parse a fixed answer-free v8 action and preserve bounded diagnostics."""

    prefix_ids = tokenizer.encode(
        "<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    completion_ids = tokenizer.encode(
        '{"name":"list_files","arguments":{}}<|im_end|>',
        add_special_tokens=False,
    )
    try:
        return parse_response_fn(tokenizer, completion_ids, prefix=prefix_ids)
    except (TypeError, ValueError) as exc:
        return {"probe_error": type(exc).__name__, "probe_message": str(exc)[:500]}


def is_valid_bare_json_tool_probe(parsed: Any) -> bool:
    """Return whether a probe result has the exact TRL tool-call shape."""

    tool_calls = parsed.get("tool_calls") if isinstance(parsed, dict) else None
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        return False
    function = tool_calls[0].get("function") if isinstance(tool_calls[0], dict) else None
    return bool(
        isinstance(function, dict)
        and function.get("name") == "list_files"
        and function.get("arguments") == {}
    )


def probe_bare_json_tool_parsing(tokenizer: Any, parse_response_fn: Any) -> bool:
    """Verify that the installed Transformers parser recognizes the v8 action shape."""

    return is_valid_bare_json_tool_probe(
        run_bare_json_tool_parsing_probe(tokenizer, parse_response_fn)
    )


if __name__ == "__main__":
    main()
