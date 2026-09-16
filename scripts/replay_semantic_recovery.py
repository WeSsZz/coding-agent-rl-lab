from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from coding_agent_rl_lab.contracts import ActionKind, AgentAction
from coding_agent_rl_lab.grpo_remote import (
    RemoteGRPOCodingEnvironment,
    add_parent_path_evidence,
)
from coding_agent_rl_lab.grpo_train import bare_json_system_prompt


class ReplayError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay real rollout prefixes and verifier-backed semantic recoveries"
    )
    parser.add_argument("--spec", required=True)
    parser.add_argument("--source-sft", required=True)
    parser.add_argument("--worker-base-url", required=True)
    parser.add_argument("--worker-token-file", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise ReplayError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)

    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    source_rows = _jsonl(Path(args.source_sft))
    source_by_task = {row["task_id"]: row for row in source_rows}
    token = Path(args.worker_token_file).read_text(encoding="utf-8").strip()
    trajectories: list[dict[str, Any]] = []
    state_reports: list[dict[str, Any]] = []

    try:
        for state in spec["states"]:
            trajectory, state_report = replay_state(
                state,
                source_by_task=source_by_task,
                worker_base_url=args.worker_base_url,
                worker_token=token,
                output_dir=output_dir,
            )
            trajectories.append(trajectory)
            state_reports.append(state_report)
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "completed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_states": state_reports,
            "spec_sha256": _sha256(spec_path),
        }
        (output_dir / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise

    trajectory_path = output_dir / "verified-trajectories.jsonl"
    trajectory_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trajectories),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "completed": True,
        "state_count": len(trajectories),
        "recovery_action_count": sum(len(row["recovery_steps"]) for row in trajectories),
        "states": state_reports,
        "spec_sha256": _sha256(spec_path),
        "trajectory_sha256": _sha256(trajectory_path),
    }
    (output_dir / "replay-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def replay_state(
    state: dict[str, Any],
    *,
    source_by_task: dict[str, dict[str, Any]],
    worker_base_url: str,
    worker_token: str,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_id = _required_text(state, "task_id")
    state_id = _required_text(state, "state_id")
    session_id = _required_text(state, "source_session_id")
    trace_path = Path(_required_text(state, "trace_path"))
    prefix_count = state.get("prefix_action_count")
    if not isinstance(prefix_count, int) or prefix_count <= 0:
        raise ReplayError(f"{state_id}: prefix_action_count must be positive")
    navigation_first = state.get("navigation_first") is True
    source = source_by_task.get(task_id)
    if source is None:
        raise ReplayError(f"{state_id}: no source SFT row for {task_id}")
    messages = source.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ReplayError(f"{state_id}: source SFT row has no initial messages")
    initial_messages = [dict(messages[0]), dict(messages[1])]
    initial_messages[0]["content"] = bare_json_system_prompt(
        initial_messages[0]["content"], navigation_first=navigation_first
    )

    session_steps = _session_steps(trace_path, task_id, session_id)
    if len(session_steps) < prefix_count:
        raise ReplayError(
            f"{state_id}: trace has {len(session_steps)} actions, needs {prefix_count}"
        )
    prefix_steps = session_steps[:prefix_count]
    expected_prefix = state.get("expected_prefix_actions")
    if expected_prefix is not None:
        actual_prefix = [step["request"]["action"] for step in prefix_steps]
        if actual_prefix != expected_prefix:
            raise ReplayError(f"{state_id}: source trace prefix disagrees with frozen spec")

    audit_path = output_dir / f"{state_id}-reward-audit.jsonl"
    environment = RemoteGRPOCodingEnvironment(
        worker_base_url,
        worker_token,
        reward_audit_path=audit_path,
        navigation_first=navigation_first,
    )
    prefix_messages: list[dict[str, Any]] = []
    recovery_steps: list[dict[str, Any]] = []
    baseline_observation = ""
    try:
        baseline_observation = environment.reset(task_id=task_id)
        for sequence, trace_step in enumerate(prefix_steps):
            action = trace_step["request"]["action"]
            observation = _execute(environment, action)
            expected_observation = str(trace_step["response"]["observation"])
            if navigation_first and action["kind"] == "read_file":
                expected_observation = add_parent_path_evidence(
                    expected_observation, action["arguments"]["path"]
                )
            _assert_observation_matches(
                state_id, sequence, action["kind"], expected_observation, observation
            )
            prefix_messages.extend(_messages_for(action, observation))
            if environment._completed:
                raise ReplayError(f"{state_id}: source prefix terminated at action {sequence}")

        teacher_actions = state.get("teacher_actions")
        if not isinstance(teacher_actions, list) or not teacher_actions:
            raise ReplayError(f"{state_id}: teacher_actions must not be empty")
        for sequence, action in enumerate(teacher_actions):
            _validate_action(action, f"{state_id} teacher action {sequence}")
            observation = _execute(environment, action)
            recovery_steps.append(
                {
                    "sequence": sequence,
                    "action": action,
                    "observation": observation,
                    "terminated": environment._completed,
                    "source_path": action.get("arguments", {}).get("path"),
                }
            )
            if environment._completed and sequence != len(teacher_actions) - 1:
                raise ReplayError(f"{state_id}: recovery terminated before its last action")
        if teacher_actions[-1]["kind"] != "run_tests" or not environment._completed:
            raise ReplayError(f"{state_id}: recovery did not terminate with successful run_tests")
        if environment.reward != 1.0:
            raise ReplayError(f"{state_id}: strict recovery reward was {environment.reward}")
    finally:
        environment._delete()

    audits = _jsonl(audit_path)
    if len(audits) != 1:
        raise ReplayError(f"{state_id}: expected one reward audit, found {len(audits)}")
    audit = audits[0]
    components = audit.get("reward_components") or {}
    expected = {
        "baseline_failure_count": 1,
        "final_failure_count": 0,
        "new_failure_count": 0,
        "strict_success": True,
    }
    for key, value in expected.items():
        if components.get(key) != value:
            raise ReplayError(
                f"{state_id}: verifier {key}={components.get(key)!r}, expected {value!r}"
            )
    if components.get("violations") not in ([], None):
        raise ReplayError(f"{state_id}: verifier reported violations")

    trajectory = {
        "schema_version": 1,
        "task_id": task_id,
        "task_set": "train",
        "state_id": state_id,
        "source_session_id": session_id,
        "source_action_count": prefix_count,
        "source_trace_sha256": _sha256(trace_path),
        "base_commit": _required_text(state, "base_commit"),
        "teacher_source": _required_text(state, "teacher_source"),
        "initial_messages": initial_messages,
        "baseline_observation": baseline_observation,
        "prefix_messages": prefix_messages,
        "recovery_steps": recovery_steps,
        "verifier": components,
    }
    state_report = {
        "state_id": state_id,
        "source_session_id": session_id,
        "source_action_count": prefix_count,
        "recovery_action_count": len(recovery_steps),
        "strict_reward": audit.get("strict_reward"),
        "source_trace_sha256": trajectory["source_trace_sha256"],
        "reward_audit_sha256": _sha256(audit_path),
    }
    return trajectory, state_report


def _session_steps(path: Path, task_id: str, session_id: str) -> list[dict[str, Any]]:
    sessions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _jsonl(path):
        request_path = row.get("path", "")
        action = row.get("request", {}).get("action")
        if not action or "/sessions/" not in request_path or not request_path.endswith("/actions"):
            continue
        parsed_session = request_path.split("/sessions/", 1)[1].split("/", 1)[0]
        sessions[(str(row.get("task_id")), parsed_session)].append(row)
    steps = sessions.get((task_id, session_id))
    if not steps:
        raise ReplayError(f"no trace actions for task/session {task_id}/{session_id}")
    return steps


def _execute(environment: RemoteGRPOCodingEnvironment, action: dict[str, Any]) -> str:
    parsed = AgentAction.from_dict(action)
    method = getattr(environment, parsed.kind.value)
    return method(**parsed.arguments)


def _messages_for(action: dict[str, Any], observation: str) -> list[dict[str, Any]]:
    call = {"name": action["kind"], "arguments": action.get("arguments", {})}
    return [
        {
            "role": "assistant",
            "content": json.dumps(call, ensure_ascii=False, separators=(",", ":")),
        },
        {"role": "tool", "name": action["kind"], "content": observation},
    ]


def _assert_observation_matches(
    state_id: str,
    sequence: int,
    kind: str,
    expected: str,
    actual: str,
) -> None:
    if kind == ActionKind.RUN_TESTS.value:
        expected_status = _pytest_status(expected)
        actual_status = _pytest_status(actual)
        if expected_status != actual_status:
            raise ReplayError(
                f"{state_id}: prefix action {sequence} verifier status changed: "
                f"{expected_status!r} != {actual_status!r}"
            )
        return
    if expected.rstrip() != actual.rstrip():
        raise ReplayError(f"{state_id}: prefix action {sequence} observation changed")


def _pytest_status(observation: str) -> tuple[str, tuple[str, ...]]:
    result = "passed" if observation.startswith("Tests passed") else "failed"
    node_ids = tuple(
        sorted(
            re.findall(
                r"(?m)^(?:FAILED|PASSED)\s+([^\s]+)",
                observation,
            )
        )
    )
    return result, node_ids


def _validate_action(action: Any, label: str) -> None:
    if not isinstance(action, dict):
        raise ReplayError(f"{label} is not an object")
    try:
        AgentAction.from_dict(action)
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayError(f"{label} is invalid") from exc
    if not isinstance(action.get("arguments", {}), dict):
        raise ReplayError(f"{label} arguments are invalid")


def _required_text(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ReplayError(f"missing non-empty {key}")
    return item


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ReplayError(f"{path} row {number} is not an object")
        rows.append(value)
    if not rows:
        raise ReplayError(f"{path} has no JSONL rows")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
