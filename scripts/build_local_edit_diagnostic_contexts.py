from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EDIT_KINDS = {"replace_text", "replace_lines"}
CALLBACK_PATH = (
    "moto/stepfunctions/parser/asl/component/state/exec/state_task/service/"
    "state_task_service_callback.py"
)


class LocalEditContextBuildError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze one-step edit and staged continuation contexts from verified trajectories"
    )
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--edit-output", required=True)
    parser.add_argument("--staged-output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    source_path = Path(args.trajectories)
    edit_output = Path(args.edit_output)
    staged_output = Path(args.staged_output)
    report_path = Path(args.report)
    for target in (edit_output, staged_output, report_path):
        if target.exists():
            raise LocalEditContextBuildError(f"refusing to overwrite existing output: {target}")

    trajectories = _jsonl(source_path)
    edit_contexts, staged_contexts = build_contexts(trajectories)
    _write_jsonl(edit_output, edit_contexts)
    _write_jsonl(staged_output, staged_contexts)
    report = {
        "schema_version": 1,
        "diagnostic_kind": "moto-7607-local-edit-and-first-error",
        "source_trajectories": str(source_path.resolve()),
        "source_sha256": _sha256(source_path),
        "source_state_ids": [row["state_id"] for row in trajectories],
        "source_trajectory_count": len(trajectories),
        "strict_source_trajectory_count": sum(
            row.get("verifier", {}).get("strict_success") is True for row in trajectories
        ),
        "edit_context_count": len(edit_contexts),
        "edit_state_ids": [row["state_id"] for row in edit_contexts],
        "edit_contexts_sha256": _sha256(edit_output),
        "staged_context_count": len(staged_contexts),
        "staged_state_ids": [row["state_id"] for row in staged_contexts],
        "staged_contexts_sha256": _sha256(staged_output),
        "selection_protocol": {
            "edit": "every replace_text/replace_lines recovery action",
            "located": "before first edit to the callback implementation",
            "before_last_edit": "before final recovery edit",
            "before_verify": "after all recovery edits and before run_tests",
        },
        "answer_leakage_check": "teacher target/remaining actions are metadata only, absent from prompt",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_contexts(
    trajectories: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(trajectories) != 3:
        raise LocalEditContextBuildError("expected exactly three verified source trajectories")
    edit_contexts: list[dict[str, Any]] = []
    staged_contexts: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    for trajectory in trajectories:
        source_id = _validate_trajectory(trajectory)
        if source_id in seen_source_ids:
            raise LocalEditContextBuildError(f"duplicate source state: {source_id}")
        seen_source_ids.add(source_id)
        source_actions, source_history = _source_prefix(trajectory)
        recovery_steps = trajectory["recovery_steps"]

        for step in recovery_steps:
            if step["action"]["kind"] not in EDIT_KINDS:
                continue
            prefix_count = int(step["sequence"])
            edit_contexts.append(
                _context(
                    trajectory,
                    state_id=f"{source_id}-edit-{prefix_count}",
                    phase="edit",
                    source_actions=source_actions,
                    source_history=source_history,
                    recovery_prefix=recovery_steps[:prefix_count],
                    metadata={
                        "target_action": step["action"],
                        "target_observation": step["observation"],
                        "teacher_recovery_sequence": prefix_count,
                    },
                )
            )

        callback_edits = [
            int(step["sequence"])
            for step in recovery_steps
            if step["action"]["kind"] in EDIT_KINDS
            and step["action"].get("arguments", {}).get("path") == CALLBACK_PATH
        ]
        all_edits = [
            int(step["sequence"])
            for step in recovery_steps
            if step["action"]["kind"] in EDIT_KINDS
        ]
        verifier_steps = [
            int(step["sequence"])
            for step in recovery_steps
            if step["action"]["kind"] == "run_tests"
        ]
        if not callback_edits or not all_edits or len(verifier_steps) != 1:
            raise LocalEditContextBuildError(f"{source_id}: staged anchors are not unique")
        anchors = {
            "located": min(callback_edits),
            "before-last-edit": max(all_edits),
            "before-verify": verifier_steps[0],
        }
        for label, prefix_count in anchors.items():
            staged_contexts.append(
                _context(
                    trajectory,
                    state_id=f"{source_id}-{label}",
                    phase=label,
                    source_actions=source_actions,
                    source_history=source_history,
                    recovery_prefix=recovery_steps[:prefix_count],
                    metadata={
                        "teacher_recovery_prefix_count": prefix_count,
                        "teacher_remaining_actions": [
                            step["action"] for step in recovery_steps[prefix_count:]
                        ],
                    },
                )
            )

    if len(edit_contexts) != 7 or len(staged_contexts) != 9:
        raise LocalEditContextBuildError(
            f"frozen protocol requires 7 edit and 9 staged contexts, got "
            f"{len(edit_contexts)} and {len(staged_contexts)}"
        )
    return edit_contexts, staged_contexts


def _context(
    trajectory: dict[str, Any],
    *,
    state_id: str,
    phase: str,
    source_actions: list[dict[str, Any]],
    source_history: list[dict[str, Any]],
    recovery_prefix: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    recovery_actions = [dict(step["action"]) for step in recovery_prefix]
    recovery_history = [
        message
        for step in recovery_prefix
        for message in (
            _assistant_message(step["action"]),
            {
                "role": "tool",
                "name": step["action"]["kind"],
                "content": step["observation"],
            },
        )
    ]
    initial = [dict(message) for message in trajectory["initial_messages"]]
    prefix_actions = [*source_actions, *recovery_actions]
    return {
        "schema_version": 1,
        "task_id": trajectory["task_id"],
        "state_id": state_id,
        "source_session_id": trajectory["source_session_id"],
        "source_state_id": trajectory["state_id"],
        "source_trace_sha256": trajectory["source_trace_sha256"],
        "base_commit": trajectory["base_commit"],
        "used_for_training": True,
        "diagnostic_phase": phase,
        "prefix_action_count": len(prefix_actions),
        "source_action_count": len(source_actions),
        "recovery_prefix_action_count": len(recovery_actions),
        "prefix_actions": prefix_actions,
        "prompt": [*initial, *source_history, *recovery_history],
        **metadata,
    }


def _source_prefix(
    trajectory: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    messages = trajectory["prefix_messages"]
    if len(messages) % 2:
        raise LocalEditContextBuildError("source prefix must contain assistant/tool pairs")
    actions: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for index in range(0, len(messages), 2):
        assistant = messages[index]
        tool = messages[index + 1]
        if assistant.get("role") != "assistant" or tool.get("role") != "tool":
            raise LocalEditContextBuildError("source prefix roles are not assistant/tool pairs")
        try:
            call = json.loads(assistant["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise LocalEditContextBuildError("invalid source assistant action") from exc
        action = {"kind": call.get("name"), "arguments": call.get("arguments")}
        _validate_action(action)
        if tool.get("name") != action["kind"] or not isinstance(tool.get("content"), str):
            raise LocalEditContextBuildError("source tool observation does not match action")
        actions.append(action)
        history.extend([_assistant_message(action), dict(tool)])
    if len(actions) != trajectory["source_action_count"]:
        raise LocalEditContextBuildError("source action count does not match prefix")
    return actions, history


def _validate_trajectory(trajectory: dict[str, Any]) -> str:
    if trajectory.get("schema_version") != 1:
        raise LocalEditContextBuildError("unsupported trajectory schema")
    if trajectory.get("task_id") != "getmoto__moto-7607":
        raise LocalEditContextBuildError("unexpected task id")
    if trajectory.get("verifier", {}).get("strict_success") is not True:
        raise LocalEditContextBuildError("source trajectory is not verifier-strict")
    state_id = trajectory.get("state_id")
    if not isinstance(state_id, str) or not state_id:
        raise LocalEditContextBuildError("invalid source state id")
    if not isinstance(trajectory.get("initial_messages"), list) or len(
        trajectory["initial_messages"]
    ) != 2:
        raise LocalEditContextBuildError(f"{state_id}: invalid initial messages")
    recovery_steps = trajectory.get("recovery_steps")
    if not isinstance(recovery_steps, list) or not recovery_steps:
        raise LocalEditContextBuildError(f"{state_id}: missing recovery steps")
    for sequence, step in enumerate(recovery_steps):
        if step.get("sequence") != sequence or not isinstance(step.get("observation"), str):
            raise LocalEditContextBuildError(f"{state_id}: invalid recovery sequence")
        _validate_action(step.get("action"))
    return state_id


def _validate_action(action: Any) -> None:
    if (
        not isinstance(action, dict)
        or not isinstance(action.get("kind"), str)
        or not isinstance(action.get("arguments"), dict)
    ):
        raise LocalEditContextBuildError("invalid action")


def _assistant_message(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": action["kind"],
                    "arguments": action["arguments"],
                },
            }
        ],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if __name__ == "__main__":
    main()
