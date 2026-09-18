"""Run teacher-positive controls and paired base/adapter edit diagnostics."""
from __future__ import annotations

import argparse
import collections
import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Any

from .contracts import ActionKind, AgentAction
from .edit_action_evaluate import _action_from_parsed
from .fixed_state_evaluate import ReplayedStateEnvironment, _load_contexts, _test_status
from .grpo_train import configure_tool_response_parsing, probe_bare_json_tool_parsing, read_worker_token
from .lora_inference import merge_lora_adapter_for_inference


class PairedEditEvaluationError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--worker-token-file", required=True)
    parser.add_argument("--worker-base-url", default="http://127.0.0.1:9017")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists():
        raise PairedEditEvaluationError(f"refusing to overwrite output directory: {output}")
    contexts_path = Path(args.contexts)
    contexts = _load_contexts(contexts_path)
    if len(contexts) != 7 or any(row.get("diagnostic_phase") != "edit" for row in contexts):
        raise PairedEditEvaluationError("paired protocol requires exactly seven edit contexts")
    trajectories_path = Path(args.trajectories)
    trajectories = _jsonl(trajectories_path)
    source_by_id = {row["state_id"]: row for row in trajectories}
    if len(source_by_id) != 3:
        raise PairedEditEvaluationError("paired protocol requires three source trajectories")
    teacher_steps = _teacher_steps(contexts, source_by_id)
    adapter = Path(args.adapter_path).resolve()
    adapter_hash_before = _sha256(adapter / "adapter_model.safetensors")
    token = read_worker_token(Path(args.worker_token_file))
    output.mkdir(parents=True)

    manifest = {
        "schema_version": 1,
        "evaluation_kind": "teacher-positive-control-and-paired-base-adapter-edit-diagnostic",
        "training_performed": False,
        "grpo_performed": False,
        "model_path": str(Path(args.model_path).resolve()),
        "adapter_path": str(adapter),
        "adapter_sha256": adapter_hash_before,
        "contexts_path": str(contexts_path.resolve()),
        "contexts_sha256": _sha256(contexts_path),
        "trajectories_path": str(trajectories_path.resolve()),
        "trajectories_sha256": _sha256(trajectories_path),
        "state_ids": [row["state_id"] for row in contexts],
        "seed": args.seed,
        "planned_teacher_trials": 14,
        "planned_model_generations": 14,
        "planned_model_execution_trials": 28,
        "budget": {
            "assistant_turns_per_state_condition": 1,
            "max_new_tokens": args.max_new_tokens,
            "decoding": "greedy",
            "conditions": ["clean-base", "p2-step12-adapter"],
        },
        "teacher_gate": "7/7 target edits apply and 7/7 teacher suffix trials are strict success",
        "semantic_success": (
            "generated edit applies and substituting it for the teacher target followed by the "
            "fixed teacher suffix yields strict success with zero new failures"
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    contexts_by_id = {row["state_id"]: row for row in contexts}
    trace_path = output / "tool-traces.jsonl"

    class TracedEnvironment(ReplayedStateEnvironment):
        trace_labels: dict[str, Any]

        def _request(self, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            response = super()._request(method, path, payload)
            record = {
                **getattr(self, "trace_labels", {}),
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
        "teacher_gate_passed": False,
        "teacher_controls": [],
        "conditions": [],
    }
    _checkpoint(output, report)

    def execute(
        *,
        context: dict[str, Any],
        condition: str,
        protocol: str,
        first_action: AgentAction | None,
        suffix_steps: list[dict[str, Any]],
        validate_teacher: bool,
    ) -> dict[str, Any]:
        state_id = context["state_id"]
        audit_path = output / f"{condition}-{state_id}-{protocol}-reward-audit.jsonl"
        environment = TracedEnvironment(
            args.worker_base_url,
            token,
            contexts_by_id=contexts_by_id,
            reward_audit_path=audit_path,
        )
        environment.trace_labels = {"condition": condition, "protocol": protocol}
        first_observation: str | None = None
        first_error: str | None = None
        first_applied = False
        suffix_records: list[dict[str, Any]] = []
        forced_verifier = False
        try:
            environment.reset(state_id=state_id, task_id=context["task_id"])
            if first_action is not None:
                try:
                    first_observation = getattr(environment, first_action.kind.value)(
                        **first_action.arguments
                    )
                    first_applied = first_action.kind in {
                        ActionKind.REPLACE_TEXT,
                        ActionKind.REPLACE_LINES,
                    } and first_observation.startswith("Updated ")
                except Exception as exc:  # tool validation is evidence; infrastructure is recorded below
                    first_error = f"{type(exc).__name__}: {str(exc)[:2000]}"
            if validate_teacher:
                expected = context["target_observation"]
                if first_observation is None or first_observation.rstrip() != expected.rstrip():
                    raise PairedEditEvaluationError(
                        f"{state_id}: teacher target observation changed"
                    )
            if protocol == "with-teacher-suffix" and not environment._completed:
                for step in suffix_steps:
                    action = AgentAction.from_dict(step["action"])
                    try:
                        observation = getattr(environment, action.kind.value)(**action.arguments)
                        step_error = None
                    except Exception as exc:  # preserve downstream incompatibility
                        observation = None
                        step_error = f"{type(exc).__name__}: {str(exc)[:2000]}"
                    matches_teacher = None
                    if validate_teacher and observation is not None:
                        if action.kind == ActionKind.RUN_TESTS:
                            matches_teacher = _test_status(observation) == _test_status(
                                step["observation"]
                            )
                        else:
                            matches_teacher = observation.rstrip() == step["observation"].rstrip()
                        if not matches_teacher:
                            raise PairedEditEvaluationError(
                                f"{state_id}: teacher suffix observation changed at "
                                f"{step['sequence']}"
                            )
                    suffix_records.append(
                        {
                            "sequence": step["sequence"],
                            "action": step["action"],
                            "observation": observation,
                            "error": step_error,
                            "matches_teacher": matches_teacher,
                        }
                    )
                    if step_error is not None or environment._completed:
                        break
            if not environment._completed and environment._session_id is not None:
                try:
                    environment.run_tests()
                    forced_verifier = True
                except Exception as exc:
                    suffix_records.append(
                        {
                            "sequence": None,
                            "action": {"kind": "run_tests", "arguments": {}},
                            "observation": None,
                            "error": f"{type(exc).__name__}: {str(exc)[:2000]}",
                            "matches_teacher": None,
                        }
                    )
            reward = environment.reward
        finally:
            environment._delete()
        audits = _jsonl(audit_path) if audit_path.exists() else []
        audit = audits[0] if len(audits) == 1 else None
        components = (audit or {}).get("reward_components") or {}
        return {
            "protocol": protocol,
            "first_action": first_action.to_dict() if first_action is not None else None,
            "first_observation": first_observation,
            "first_error": first_error,
            "first_edit_applied": first_applied,
            "suffix_records": suffix_records,
            "forced_verifier": forced_verifier,
            "reward": reward,
            "audit_record_count": len(audits),
            "strict_success": (audit or {}).get("strict_reward") == 1.0,
            "resolved_failure_count": components.get("resolved_failure_count"),
            "new_failure_count": components.get("new_failure_count"),
            "reward_components": (audit or {}).get("reward_components"),
        }

    # C0: answer-bearing positive controls run before any model is loaded.
    for context in contexts:
        source_steps, target_step, suffix = teacher_steps[context["state_id"]]
        del source_steps
        target_action = AgentAction.from_dict(target_step["action"])
        immediate = execute(
            context=context,
            condition="teacher",
            protocol="immediate",
            first_action=target_action,
            suffix_steps=[],
            validate_teacher=True,
        )
        with_suffix = execute(
            context=context,
            condition="teacher",
            protocol="with-teacher-suffix",
            first_action=target_action,
            suffix_steps=suffix,
            validate_teacher=True,
        )
        report["teacher_controls"].append(
            {
                "state_id": context["state_id"],
                "source_state_id": context["source_state_id"],
                "teacher_recovery_sequence": context["teacher_recovery_sequence"],
                "target_action": target_step["action"],
                "suffix_action_count": len(suffix),
                "immediate": immediate,
                "with_teacher_suffix": with_suffix,
            }
        )
        _checkpoint(output, report)
    teacher_applied = sum(
        row["immediate"]["first_edit_applied"] for row in report["teacher_controls"]
    )
    teacher_suffix_strict = sum(
        row["with_teacher_suffix"]["strict_success"] for row in report["teacher_controls"]
    )
    teacher_suffix_new_failures = sum(
        (row["with_teacher_suffix"]["new_failure_count"] or 0)
        for row in report["teacher_controls"]
    )
    report["teacher_summary"] = {
        "state_count": len(contexts),
        "target_edit_applied_count": teacher_applied,
        "immediate_strict_success_count": sum(
            row["immediate"]["strict_success"] for row in report["teacher_controls"]
        ),
        "immediate_failure_reduction_count": sum(
            (row["immediate"]["resolved_failure_count"] or 0) > 0
            for row in report["teacher_controls"]
        ),
        "suffix_strict_success_count": teacher_suffix_strict,
        "suffix_new_failure_count_total": teacher_suffix_new_failures,
    }
    report["teacher_gate_passed"] = (
        teacher_applied == 7
        and teacher_suffix_strict == 7
        and teacher_suffix_new_failures == 0
    )
    _checkpoint(output, report)
    if not report["teacher_gate_passed"]:
        raise PairedEditEvaluationError("teacher positive-control gate failed")

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
        raise PairedEditEvaluationError("bare JSON parser probe failed")
    schema_environment = ReplayedStateEnvironment(
        "http://127.0.0.1:1",
        "schema-only-token",
        contexts_by_id=contexts_by_id,
        reward_audit_path=output / "unused-schema-audit.jsonl",
    )
    methods = [
        member
        for name, member in inspect.getmembers(schema_environment, predicate=inspect.ismethod)
        if name not in {"reset", "get_reward"} and not name.startswith("_")
    ]
    tools = [get_json_schema(method) for method in methods]
    prefix_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    supplied_eos_ids = _as_id_set(tokenizer.eos_token_id)
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    model_path = Path(args.model_path)
    report["runtime"] = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "tokenizer_vocab_size": len(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "all_special_tokens": tokenizer.all_special_tokens,
        "all_special_ids": tokenizer.all_special_ids,
        "chat_template_sha256": _sha256_text(str(tokenizer.chat_template)),
        "model_config_sha256": _optional_sha256(model_path / "config.json"),
        "tokenizer_config_sha256": _optional_sha256(model_path / "tokenizer_config.json"),
        "special_tokens_map_sha256": _optional_sha256(model_path / "special_tokens_map.json"),
        "tools_schema_sha256": _sha256_text(json.dumps(tools, ensure_ascii=False, sort_keys=True)),
    }
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, local_files_only=True
    )
    model.to("cuda:0")
    model.requires_grad_(False)
    model.eval()

    def run_condition(condition: str, *, adapter_loaded: bool) -> dict[str, Any]:
        condition_report: dict[str, Any] = {
            "condition": condition,
            "adapter_loaded": adapter_loaded,
            "adapter_sha256": adapter_hash_before if adapter_loaded else None,
            "model_has_peft_config": hasattr(model, "peft_config"),
            "states": [],
        }
        generation_path = output / f"{condition}-generations.jsonl"
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
            prompt_ids = model_inputs["input_ids"][0].tolist()
            with torch.inference_mode():
                generated = model.generate(
                    **model_inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completion_ids = generated[0, len(prompt_ids):].tolist()
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
            parse_result: Any = None
            parse_error: str | None = None
            try:
                parse_result = tokenizer.parse_response(completion_ids, prefix=prefix_ids)
                action = _action_from_parsed(parse_result)
            except Exception as exc:
                action = None
                parse_error = f"{type(exc).__name__}: {str(exc)[:2000]}"
            _, _, suffix = teacher_steps[state_id]
            immediate = execute(
                context=context,
                condition=condition,
                protocol="immediate",
                first_action=action,
                suffix_steps=[],
                validate_teacher=False,
            )
            hybrid = execute(
                context=context,
                condition=condition,
                protocol="with-teacher-suffix",
                first_action=action,
                suffix_steps=suffix,
                validate_teacher=False,
            )
            special = _special_token_stats(
                completion_ids,
                all_special_ids=set(tokenizer.all_special_ids),
                eos_ids=supplied_eos_ids,
                tokenizer=tokenizer,
            )
            generated_action = action.to_dict() if action is not None else None
            teacher_action = context["target_action"]
            semantic_success = bool(
                immediate["first_edit_applied"]
                and hybrid["strict_success"]
                and (hybrid["new_failure_count"] or 0) == 0
            )
            entry = {
                "state_id": state_id,
                "source_state_id": context["source_state_id"],
                "teacher_recovery_sequence": context["teacher_recovery_sequence"],
                "seed": args.seed,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "prompt_token_count": len(prompt_ids),
                "prompt_token_ids_sha256": _sha256_text(json.dumps(prompt_ids)),
                "completion_token_count": len(completion_ids),
                "completion_token_ids": completion_ids,
                "completion_text": completion_text,
                "termination_reason": _termination_reason(
                    completion_ids,
                    eos_ids=supplied_eos_ids,
                    max_new_tokens=args.max_new_tokens,
                ),
                "special_tokens": special,
                "special_token_degeneration": bool(
                    action is None
                    and completion_ids
                    and special["special_token_fraction"] >= 0.5
                ),
                "parse_result": parse_result,
                "parse_error": parse_error,
                "generated_action": generated_action,
                "teacher_target_action": teacher_action,
                "exact_teacher_action_match": generated_action == teacher_action,
                "teacher_tool_kind_match": (
                    generated_action is not None
                    and generated_action["kind"] == teacher_action["kind"]
                ),
                "immediate": immediate,
                "with_teacher_suffix": hybrid,
                "execution_semantic_success": semantic_success,
            }
            condition_report["states"].append(entry)
            with generation_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            _checkpoint(output, report)
        condition_report["summary"] = _condition_summary(condition_report["states"])
        return condition_report

    try:
        base = run_condition("clean-base", adapter_loaded=False)
        report["conditions"].append(base)
        _checkpoint(output, report)
        adapter_merge = merge_lora_adapter_for_inference(model, adapter)
        report["adapter_merge"] = adapter_merge
        adapted = run_condition("p2-step12-adapter", adapter_loaded=True)
        report["conditions"].append(adapted)
        report["paired_comparison"] = _paired_comparison(base, adapted)
        report["adapter_unchanged"] = adapter_hash_before == _sha256(
            adapter / "adapter_model.safetensors"
        )
        report["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        if not report["adapter_unchanged"]:
            raise PairedEditEvaluationError("evaluation mutated adapter weights")
        report["run_complete"] = True
        _checkpoint(output, report)
        print(
            json.dumps(
                {
                    "teacher": report["teacher_summary"],
                    "clean_base": base["summary"],
                    "adapter": adapted["summary"],
                    "paired": report["paired_comparison"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        schema_environment._delete()


def _teacher_steps(
    contexts: list[dict[str, Any]], source_by_id: dict[str, dict[str, Any]]
) -> dict[str, tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]]:
    result = {}
    for context in contexts:
        source_id = context["source_state_id"]
        trajectory = source_by_id.get(source_id)
        if trajectory is None or trajectory.get("verifier", {}).get("strict_success") is not True:
            raise PairedEditEvaluationError(f"{source_id}: missing strict source trajectory")
        steps = trajectory["recovery_steps"]
        sequence = context["teacher_recovery_sequence"]
        if sequence >= len(steps) or steps[sequence]["action"] != context["target_action"]:
            raise PairedEditEvaluationError(f"{context['state_id']}: teacher target mismatch")
        if steps[sequence]["observation"] != context["target_observation"]:
            raise PairedEditEvaluationError(f"{context['state_id']}: teacher observation mismatch")
        result[context["state_id"]] = (steps[:sequence], steps[sequence], steps[sequence + 1 :])
    return result


def _termination_reason(
    completion_ids: list[int], *, eos_ids: set[int], max_new_tokens: int
) -> str:
    if completion_ids and completion_ids[-1] in eos_ids:
        return "eos_token"
    if len(completion_ids) >= max_new_tokens:
        return "max_new_tokens"
    if not completion_ids:
        return "empty_generation"
    return "stopped_without_supplied_eos"


def _special_token_stats(
    completion_ids: list[int], *, all_special_ids: set[int], eos_ids: set[int], tokenizer: Any
) -> dict[str, Any]:
    special_ids = [token_id for token_id in completion_ids if token_id in all_special_ids]
    counts = collections.Counter(special_ids)
    return {
        "special_token_count": len(special_ids),
        "nonterminal_special_token_count": sum(
            count for token_id, count in counts.items() if token_id not in eos_ids
        ),
        "special_token_fraction": (
            len(special_ids) / len(completion_ids) if completion_ids else 0.0
        ),
        "counts": [
            {
                "token_id": token_id,
                "token": tokenizer.convert_ids_to_tokens(token_id),
                "count": count,
            }
            for token_id, count in sorted(counts.items())
        ],
    }


def _condition_summary(states: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "state_count": len(states),
        "parsed_action_count": sum(row["generated_action"] is not None for row in states),
        "edit_action_count": sum(
            row["generated_action"] is not None
            and row["generated_action"]["kind"] in {"replace_text", "replace_lines"}
            for row in states
        ),
        "edit_applied_count": sum(row["immediate"]["first_edit_applied"] for row in states),
        "immediate_strict_success_count": sum(row["immediate"]["strict_success"] for row in states),
        "immediate_failure_reduction_count": sum(
            (row["immediate"]["resolved_failure_count"] or 0) > 0 for row in states
        ),
        "teacher_suffix_strict_success_count": sum(
            row["with_teacher_suffix"]["strict_success"] for row in states
        ),
        "execution_semantic_success_count": sum(
            row["execution_semantic_success"] for row in states
        ),
        "special_token_degeneration_count": sum(
            row["special_token_degeneration"] for row in states
        ),
        "max_new_tokens_count": sum(
            row["termination_reason"] == "max_new_tokens" for row in states
        ),
        "exact_teacher_action_match_count": sum(
            row["exact_teacher_action_match"] for row in states
        ),
    }


def _paired_comparison(base: dict[str, Any], adapted: dict[str, Any]) -> dict[str, Any]:
    adapted_by_id = {row["state_id"]: row for row in adapted["states"]}
    rows = []
    for base_row in base["states"]:
        adapted_row = adapted_by_id[base_row["state_id"]]
        rows.append(
            {
                "state_id": base_row["state_id"],
                "base_parsed": base_row["generated_action"] is not None,
                "adapter_parsed": adapted_row["generated_action"] is not None,
                "base_special_token_degeneration": base_row["special_token_degeneration"],
                "adapter_special_token_degeneration": adapted_row["special_token_degeneration"],
                "base_edit_applied": base_row["immediate"]["first_edit_applied"],
                "adapter_edit_applied": adapted_row["immediate"]["first_edit_applied"],
                "base_execution_semantic_success": base_row["execution_semantic_success"],
                "adapter_execution_semantic_success": adapted_row["execution_semantic_success"],
                "same_completion_token_ids": (
                    base_row["completion_token_ids"] == adapted_row["completion_token_ids"]
                ),
            }
        )
    return {
        "states": rows,
        "base_semantic_wins": sum(
            row["base_execution_semantic_success"]
            and not row["adapter_execution_semantic_success"]
            for row in rows
        ),
        "adapter_semantic_wins": sum(
            row["adapter_execution_semantic_success"]
            and not row["base_execution_semantic_success"]
            for row in rows
        ),
        "base_fewer_special_token_degenerations": (
            base["summary"]["special_token_degeneration_count"]
            < adapted["summary"]["special_token_degeneration_count"]
        ),
    }


def _as_id_set(value: Any) -> set[int]:
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value}
    return set()


def _optional_sha256(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _checkpoint(output: Path, report: dict[str, Any]) -> None:
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")


if __name__ == "__main__":
    main()
