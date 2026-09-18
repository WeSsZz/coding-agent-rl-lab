"""Generate and execute one free next action from frozen real pre-edit states."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Any

from .contracts import ActionKind, AgentAction
from .fixed_state_evaluate import FixedStateEvaluationError, ReplayedStateEnvironment, _load_contexts
from .grpo_train import configure_tool_response_parsing, probe_bare_json_tool_parsing, read_worker_token
from .lora_inference import merge_lora_adapter_for_inference


class EditActionEvaluationError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--worker-token-file", required=True)
    parser.add_argument("--worker-base-url", default="http://127.0.0.1:9016")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists():
        raise EditActionEvaluationError(f"refusing to overwrite output directory: {output}")
    contexts_path = Path(args.contexts)
    contexts = _load_contexts(contexts_path)
    if len(contexts) != 7 or any(row.get("diagnostic_phase") != "edit" for row in contexts):
        raise EditActionEvaluationError("one-step protocol requires exactly seven edit contexts")
    adapter = Path(args.adapter_path).resolve()
    adapter_hash_before = _sha256(adapter / "adapter_model.safetensors")
    token = read_worker_token(Path(args.worker_token_file))
    output.mkdir(parents=True)

    manifest = {
        "schema_version": 1,
        "evaluation_kind": "frozen-real-pre-edit-one-step-diagnostic",
        "training_performed": False,
        "model_path": str(Path(args.model_path).resolve()),
        "adapter_path": str(adapter),
        "adapter_sha256": adapter_hash_before,
        "contexts_path": str(contexts_path.resolve()),
        "contexts_sha256": _sha256(contexts_path),
        "state_ids": [row["state_id"] for row in contexts],
        "seed": args.seed,
        "planned_trial_count": len(contexts),
        "budget": {
            "assistant_turns_per_state": 1,
            "max_new_tokens": args.max_new_tokens,
            "decoding": "greedy",
        },
        "success_definition": (
            "generated edit parses, applies, and the complete official verifier reports strict success; "
            "exact teacher-string equality is auxiliary only"
        ),
        "stop_gate": "zero verifier-improving edits stops before data expansion or training",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from transformers.utils import get_json_schema

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    configure_tool_response_parsing(tokenizer, bare_json_tool_calls=True)
    if not probe_bare_json_tool_parsing(
        tokenizer, lambda active, ids, *, prefix: active.parse_response(ids, prefix=prefix)
    ):
        raise EditActionEvaluationError("bare JSON parser probe failed")
    schema_environment = ReplayedStateEnvironment(
        "http://127.0.0.1:1",
        "schema-only-token",
        contexts_by_id={row["state_id"]: row for row in contexts},
        reward_audit_path=output / "unused-schema-audit.jsonl",
    )
    methods = [
        member
        for name, member in inspect.getmembers(schema_environment, predicate=inspect.ismethod)
        if name not in {"reset", "get_reward"} and not name.startswith("_")
    ]
    tools = [get_json_schema(method) for method in methods]
    prefix_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, local_files_only=True
    )
    model.to("cuda:0")
    adapter_merge = merge_lora_adapter_for_inference(model, adapter)
    model.requires_grad_(False)
    model.eval()

    contexts_by_id = {row["state_id"]: row for row in contexts}
    trace_path = output / "tool-traces.jsonl"

    class TracedEnvironment(ReplayedStateEnvironment):
        def _request(self, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            response = super()._request(method, path, payload)
            record = {
                "state_id": self.state_id,
                "task_id": payload.get("task_id", self._task_id),
                "method": method,
                "path": path,
                "request": payload,
                "response": response,
            }
            with trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            return response

    report: dict[str, Any] = {
        **manifest,
        "run_complete": False,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "adapter_merge": adapter_merge,
        "states": [],
    }
    _checkpoint(output, report)
    try:
        for context in contexts:
            state_id = context["state_id"]
            set_seed(args.seed)
            started = time.monotonic()
            encoded = tokenizer.apply_chat_template(
                context["prompt"],
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            model_inputs = {key: value.to("cuda:0") for key, value in encoded.items()}
            prompt_length = int(model_inputs["input_ids"].shape[-1])
            with torch.inference_mode():
                generated = model.generate(
                    **model_inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completion_ids = generated[0, prompt_length:].tolist()
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
            parse_result: Any
            parse_error: str | None = None
            try:
                parse_result = tokenizer.parse_response(completion_ids, prefix=prefix_ids)
                action = _action_from_parsed(parse_result)
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                parse_result = None
                action = None
                parse_error = f"{type(exc).__name__}: {str(exc)[:1000]}"

            audit_path = output / f"{state_id}-reward-audit.jsonl"
            environment = TracedEnvironment(
                args.worker_base_url,
                token,
                contexts_by_id=contexts_by_id,
                reward_audit_path=audit_path,
            )
            observation: str | None = None
            execution_error: str | None = None
            applied = False
            verifier_forced_after_applied_edit = False
            try:
                environment.reset(state_id=state_id, task_id=context["task_id"])
                if action is not None:
                    try:
                        observation = getattr(environment, action.kind.value)(**action.arguments)
                        applied = action.kind in {
                            ActionKind.REPLACE_TEXT,
                            ActionKind.REPLACE_LINES,
                        } and observation.startswith("Updated ")
                    except (TypeError, ValueError, FixedStateEvaluationError) as exc:
                        execution_error = f"{type(exc).__name__}: {str(exc)[:1000]}"
                if applied and not environment._completed:
                    environment.run_tests()
                    verifier_forced_after_applied_edit = True
                reward = environment.reward
            finally:
                environment._delete()
            audit_records = _jsonl(audit_path)
            if len(audit_records) != 1:
                raise EditActionEvaluationError(
                    f"{state_id}: expected one reward audit, found {len(audit_records)}"
                )
            audit = audit_records[0]
            reward_components = audit.get("reward_components") or {}
            generated_action = action.to_dict() if action is not None else None
            target_action = context["target_action"]
            entry = {
                "state_id": state_id,
                "source_state_id": context["source_state_id"],
                "teacher_recovery_sequence": context["teacher_recovery_sequence"],
                "seed": args.seed,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "prompt_token_count": prompt_length,
                "completion_token_count": len(completion_ids),
                "hit_max_new_tokens": len(completion_ids) >= args.max_new_tokens,
                "completion_text": completion_text,
                "parse_result": parse_result,
                "parse_error": parse_error,
                "generated_action": generated_action,
                "teacher_target_action": target_action,
                "exact_teacher_action_match": generated_action == target_action,
                "teacher_tool_kind_match": (
                    generated_action is not None
                    and generated_action["kind"] == target_action["kind"]
                ),
                "parameter_validation_passed": action is not None and execution_error is None,
                "execution_error": execution_error,
                "observation": observation,
                "edit_applied": applied,
                "target_file_edit_applied": (
                    applied
                    and action is not None
                    and action.arguments.get("path")
                    == target_action.get("arguments", {}).get("path")
                ),
                "verifier_forced_after_applied_edit": verifier_forced_after_applied_edit,
                "reward": reward,
                "strict_reward": audit["strict_reward"],
                "reward_components": audit.get("reward_components"),
                "verifier_improving_edit": bool(
                    applied
                    and reward_components.get("resolved_failure_count", 0) > 0
                    and reward_components.get("new_failure_count", 0) == 0
                ),
                "strict_success": audit.get("strict_reward") == 1.0,
            }
            report["states"].append(entry)
            with (output / "generations.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            report["completed_trial_count"] = len(report["states"])
            _checkpoint(output, report)

        report["summary"] = {
            "trial_count": len(report["states"]),
            "parsed_action_count": sum(row["generated_action"] is not None for row in report["states"]),
            "edit_action_count": sum(
                row["generated_action"] is not None
                and row["generated_action"]["kind"] in {"replace_text", "replace_lines"}
                for row in report["states"]
            ),
            "edit_applied_count": sum(row["edit_applied"] for row in report["states"]),
            "target_file_edit_applied_count": sum(
                row["target_file_edit_applied"] for row in report["states"]
            ),
            "verifier_improving_edit_count": sum(
                row["verifier_improving_edit"] for row in report["states"]
            ),
            "strict_success_count": sum(row["strict_success"] for row in report["states"]),
            "exact_teacher_action_match_count": sum(
                row["exact_teacher_action_match"] for row in report["states"]
            ),
        }
        report["adapter_unchanged"] = adapter_hash_before == _sha256(
            adapter / "adapter_model.safetensors"
        )
        report["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        if not report["adapter_unchanged"]:
            raise EditActionEvaluationError("evaluation mutated adapter weights")
        report["run_complete"] = True
        _checkpoint(output, report)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    finally:
        schema_environment._delete()


def _action_from_parsed(parsed: Any) -> AgentAction:
    if not isinstance(parsed, dict):
        raise ValueError("parsed response is not an object")
    tool_calls = parsed.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise ValueError("parsed response does not contain exactly one tool call")
    function = tool_calls[0].get("function") if isinstance(tool_calls[0], dict) else None
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise ValueError("parsed tool call has no function name")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("parsed tool arguments are not an object")
    return AgentAction.from_dict({"kind": function["name"], "arguments": arguments})


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _checkpoint(output: Path, report: dict[str, Any]) -> None:
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")


if __name__ == "__main__":
    main()
