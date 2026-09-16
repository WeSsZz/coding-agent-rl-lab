from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .sft_grpo import GRPO_ACTION_PROTOCOL, GRPO_SFT_PROMPT_VERSION, GRPO_SFT_SCHEMA
from .swe_gym_smoke import pinned_rows_for_task_set


SEMANTIC_RECOVERY_ANSWER_SOURCE = "verified-train-gold-semantic-recovery"
MIXED_AUDITED_ANSWER_SOURCE = "mixed-audited-train-supervision"
SEMANTIC_RECOVERY_STAGE = "semantic-recovery"


class SemanticRecoveryError(ValueError):
    pass


def build_semantic_recovery_examples(
    trajectories: Iterable[dict[str, Any]],
    *,
    minimum_state_count: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if minimum_state_count <= 0:
        raise SemanticRecoveryError("minimum_state_count must be positive")
    allowed_tasks = {item.instance_id for item in pinned_rows_for_task_set("train")}
    output: list[dict[str, Any]] = []
    seen_examples: set[str] = set()
    seen_states: set[str] = set()
    task_ids: set[str] = set()
    state_counts: dict[str, int] = {}
    source_hashes: set[str] = set()

    for index, trajectory in enumerate(trajectories, start=1):
        _validate_trajectory(trajectory, index, allowed_tasks)
        state_id = trajectory["state_id"]
        if state_id in seen_states:
            raise SemanticRecoveryError(f"duplicate semantic recovery state_id: {state_id}")
        seen_states.add(state_id)
        task_ids.add(trajectory["task_id"])
        source_hashes.add(trajectory["source_trace_sha256"])

        history = [dict(message) for message in trajectory["initial_messages"]]
        history.extend(dict(message) for message in trajectory["prefix_messages"])
        state_count = 0
        for sequence, step in enumerate(trajectory["recovery_steps"]):
            action = step["action"]
            tool_call = {"name": action["kind"], "arguments": action["arguments"]}
            assistant = {
                "role": "assistant",
                "content": json.dumps(tool_call, ensure_ascii=False, separators=(",", ":")),
            }
            identity_payload = {
                "state_id": state_id,
                "sequence": sequence,
                "history": history,
                "target": tool_call,
            }
            identity = json.dumps(
                identity_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            example_id = "semantic-recovery-sft-" + hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()[:20]
            if example_id in seen_examples:
                raise SemanticRecoveryError(f"duplicate semantic recovery example: {example_id}")
            seen_examples.add(example_id)
            output.append(
                {
                    "schema_version": 1,
                    "example_id": example_id,
                    "task_id": trajectory["task_id"],
                    "task_set": "train",
                    "stage": SEMANTIC_RECOVERY_STAGE,
                    "source_path": step.get("source_path"),
                    "prompt_version": GRPO_SFT_PROMPT_VERSION,
                    "messages": [*history, assistant],
                    "target_action": dict(action),
                    "contains_answers": True,
                    "answer_source": SEMANTIC_RECOVERY_ANSWER_SOURCE,
                    "action_protocol": GRPO_ACTION_PROTOCOL,
                    "target_tool_call": tool_call,
                    "semantic_recovery": {
                        "state_id": state_id,
                        "sequence": sequence,
                        "source_session_id": trajectory["source_session_id"],
                        "source_action_count": trajectory["source_action_count"],
                        "source_trace_sha256": trajectory["source_trace_sha256"],
                        "base_commit": trajectory["base_commit"],
                        "teacher_source": trajectory["teacher_source"],
                        "verified_strict_success": True,
                    },
                }
            )
            history.extend(
                (
                    assistant,
                    {
                        "role": "tool",
                        "name": action["kind"],
                        "content": step["observation"],
                    },
                )
            )
            state_count += 1
        state_counts[state_id] = state_count

    if len(seen_states) < minimum_state_count:
        raise SemanticRecoveryError(
            f"semantic recovery requires at least {minimum_state_count} distinct verified states"
        )
    if not output:
        raise SemanticRecoveryError("semantic recovery dataset must not be empty")
    report = {
        "schema_version": 1,
        "dataset_schema": GRPO_SFT_SCHEMA,
        "task_set": "train",
        "task_ids": sorted(task_ids),
        "task_count": len(task_ids),
        "example_count": len(output),
        "state_count": len(seen_states),
        "state_example_counts": state_counts,
        "stage_counts": {SEMANTIC_RECOVERY_STAGE: len(output)},
        "contains_answers": True,
        "answer_source": SEMANTIC_RECOVERY_ANSWER_SOURCE,
        "prompt_version": GRPO_SFT_PROMPT_VERSION,
        "action_protocol": GRPO_ACTION_PROTOCOL,
        "trajectory_layout": "verified-real-state-semantic-recovery",
        "source_trace_sha256": sorted(source_hashes),
        "training_performed": False,
        "intended_use": "train-split-only-completion-loss-semantic-recovery",
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train-only completion examples from verified semantic recovery trajectories"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--minimum-state-count", type=int, default=3)
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    for target in (output_path, report_path):
        if target.exists():
            raise SemanticRecoveryError(f"refusing to overwrite existing output: {target}")
    trajectories = _jsonl(input_path)
    examples, report = build_semantic_recovery_examples(
        trajectories, minimum_state_count=args.minimum_state_count
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples),
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _validate_trajectory(
    trajectory: Any,
    index: int,
    allowed_tasks: set[str],
) -> None:
    if not isinstance(trajectory, dict) or trajectory.get("schema_version") != 1:
        raise SemanticRecoveryError(f"trajectory {index} has invalid schema")
    if trajectory.get("task_set") != "train" or trajectory.get("task_id") not in allowed_tasks:
        raise SemanticRecoveryError(f"trajectory {index} is outside the train split")
    for field in ("state_id", "source_session_id", "base_commit", "teacher_source"):
        if not isinstance(trajectory.get(field), str) or not trajectory[field]:
            raise SemanticRecoveryError(f"trajectory {index} requires {field}")
    source_hash = trajectory.get("source_trace_sha256")
    if not isinstance(source_hash, str) or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None:
        raise SemanticRecoveryError(f"trajectory {index} has invalid source_trace_sha256")
    if not isinstance(trajectory.get("source_action_count"), int) or trajectory["source_action_count"] < 0:
        raise SemanticRecoveryError(f"trajectory {index} has invalid source_action_count")
    initial = trajectory.get("initial_messages")
    if not isinstance(initial, list) or [message.get("role") for message in initial] != [
        "system",
        "user",
    ]:
        raise SemanticRecoveryError(f"trajectory {index} has invalid initial messages")
    _validate_text_messages(initial, f"trajectory {index} initial messages")
    prefix = trajectory.get("prefix_messages")
    if not isinstance(prefix, list) or len(prefix) % 2:
        raise SemanticRecoveryError(f"trajectory {index} has incomplete prefix history")
    _validate_history(prefix, f"trajectory {index} prefix")
    steps = trajectory.get("recovery_steps")
    if not isinstance(steps, list) or not steps:
        raise SemanticRecoveryError(f"trajectory {index} requires recovery steps")
    for step_number, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise SemanticRecoveryError(f"trajectory {index} step {step_number} is invalid")
        _validate_action(step.get("action"), f"trajectory {index} step {step_number}")
        if not isinstance(step.get("observation"), str) or not step["observation"]:
            raise SemanticRecoveryError(
                f"trajectory {index} step {step_number} lacks a real observation"
            )
    if steps[-1]["action"]["kind"] != "run_tests" or steps[-1].get("terminated") is not True:
        raise SemanticRecoveryError(f"trajectory {index} must terminate with run_tests")
    verifier = trajectory.get("verifier")
    if not isinstance(verifier, dict):
        raise SemanticRecoveryError(f"trajectory {index} requires verifier evidence")
    if (
        verifier.get("strict_success") is not True
        or verifier.get("baseline_failure_count") != 1
        or verifier.get("final_failure_count") != 0
        or verifier.get("new_failure_count") != 0
        or verifier.get("violations") != []
    ):
        raise SemanticRecoveryError(f"trajectory {index} is not a verified strict recovery")


def _validate_history(messages: list[Any], label: str) -> None:
    _validate_text_messages(messages, label)
    expected = ["assistant", "tool"] * (len(messages) // 2)
    if [message.get("role") for message in messages] != expected:
        raise SemanticRecoveryError(f"{label} roles are invalid")
    for assistant, tool in zip(messages[::2], messages[1::2], strict=True):
        try:
            call = json.loads(assistant["content"])
        except json.JSONDecodeError as exc:
            raise SemanticRecoveryError(f"{label} contains invalid assistant JSON") from exc
        if (
            not isinstance(call, dict)
            or call.get("name") != tool.get("name")
            or not isinstance(call.get("arguments"), dict)
        ):
            raise SemanticRecoveryError(f"{label} action disagrees with its tool result")


def _validate_text_messages(messages: list[Any], label: str) -> None:
    if any(
        not isinstance(message, dict) or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise SemanticRecoveryError(f"{label} contains an invalid message")


def _validate_action(action: Any, label: str) -> None:
    if not isinstance(action, dict) or not isinstance(action.get("arguments"), dict):
        raise SemanticRecoveryError(f"{label} has invalid action")
    kind = action.get("kind")
    arguments = action["arguments"]
    if kind == "search_text":
        valid = isinstance(arguments.get("query"), str) and bool(arguments["query"])
    elif kind == "read_file":
        valid = isinstance(arguments.get("path"), str) and bool(arguments["path"])
    elif kind == "replace_text":
        valid = all(isinstance(arguments.get(key), str) for key in ("path", "old", "new")) and bool(
            arguments.get("path")
        ) and bool(arguments.get("old"))
    elif kind == "replace_lines":
        valid = (
            isinstance(arguments.get("path"), str)
            and bool(arguments["path"])
            and isinstance(arguments.get("start_line"), int)
            and isinstance(arguments.get("end_line"), int)
            and arguments["start_line"] > 0
            and arguments["end_line"] >= arguments["start_line"]
            and isinstance(arguments.get("new"), str)
        )
    elif kind in {"run_tests", "finish", "list_files"}:
        valid = not arguments
    else:
        valid = False
    if not valid:
        raise SemanticRecoveryError(f"{label} has unsupported or incomplete {kind!r} action")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SemanticRecoveryError(f"invalid JSON on input row {line_number}") from exc
        if not isinstance(row, dict):
            raise SemanticRecoveryError(f"input row {line_number} must be an object")
        rows.append(row)
    if not rows:
        raise SemanticRecoveryError("semantic recovery input must not be empty")
    return rows


if __name__ == "__main__":
    main()
