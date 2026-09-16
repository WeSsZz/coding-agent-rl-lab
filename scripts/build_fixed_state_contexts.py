from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from coding_agent_rl_lab.contracts import AgentAction
from coding_agent_rl_lab.grpo_remote import RemoteGRPOCodingEnvironment, add_parent_path_evidence
from coding_agent_rl_lab.grpo_train import bare_json_system_prompt


class ContextBuildError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay and freeze real fixed-state evaluation contexts")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--source-sft", required=True)
    parser.add_argument("--worker-base-url", required=True)
    parser.add_argument("--worker-token-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    report_path = Path(args.report)
    for target in (output, report_path):
        if target.exists():
            raise ContextBuildError(f"refusing to overwrite existing output: {target}")

    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    source_rows = _jsonl(Path(args.source_sft))
    source_by_task = {row["task_id"]: row for row in source_rows}
    token = Path(args.worker_token_file).read_text(encoding="utf-8").strip()
    contexts: list[dict[str, Any]] = []
    for state in spec["states"]:
        contexts.append(
            build_context(
                state,
                source_by_task=source_by_task,
                token=token,
                worker_base_url=args.worker_base_url,
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in contexts),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "context_count": len(contexts),
        "task_ids": sorted({row["task_id"] for row in contexts}),
        "state_ids": [row["state_id"] for row in contexts],
        "training_state_count": sum(row["used_for_training"] for row in contexts),
        "unseen_session_state_count": sum(not row["used_for_training"] for row in contexts),
        "diagnostic_kind": (
            "training-state fit check plus one unseen navigation state; "
            "not a held-out edit-only set"
        ),
        "spec_sha256": _sha256(spec_path),
        "contexts_sha256": _sha256(output),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_context(
    state: dict[str, Any],
    *,
    source_by_task: dict[str, dict[str, Any]],
    token: str,
    worker_base_url: str,
) -> dict[str, Any]:
    task_id = str(state["task_id"])
    state_id = str(state["state_id"])
    session_id = str(state["source_session_id"])
    prefix_count = int(state["prefix_action_count"])
    trace_path = Path(state["trace_path"])
    steps = _session_steps(trace_path, task_id, session_id)[:prefix_count]
    if len(steps) != prefix_count:
        raise ContextBuildError(f"{state_id}: incomplete source prefix")
    source = source_by_task[task_id]
    initial = [dict(source["messages"][0]), dict(source["messages"][1])]
    initial[0]["content"] = bare_json_system_prompt(
        initial[0]["content"], navigation_first=True
    )
    environment = RemoteGRPOCodingEnvironment(worker_base_url, token, navigation_first=True)
    prefix_messages: list[dict[str, Any]] = []
    prefix_actions: list[dict[str, Any]] = []
    try:
        baseline = environment.reset(task_id=task_id)
        for sequence, step in enumerate(steps):
            action = step["request"]["action"]
            observation = _execute(environment, action)
            expected = str(step["response"]["observation"])
            if action["kind"] == "read_file":
                expected = add_parent_path_evidence(expected, action["arguments"]["path"])
            if action["kind"] == "run_tests":
                if _test_status(expected) != _test_status(observation):
                    raise ContextBuildError(f"{state_id}: verifier prefix changed at {sequence}")
            elif expected.rstrip() != observation.rstrip():
                raise ContextBuildError(f"{state_id}: prefix observation changed at {sequence}")
            if environment._completed:
                raise ContextBuildError(f"{state_id}: prefix terminated at {sequence}")
            prefix_actions.append(action)
            prefix_messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {"name": action["kind"], "arguments": action["arguments"]},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                    {"role": "tool", "name": action["kind"], "content": observation},
                ]
            )
    finally:
        environment._delete()
    return {
        "schema_version": 1,
        "task_id": task_id,
        "state_id": state_id,
        "source_session_id": session_id,
        "source_trace_sha256": _sha256(trace_path),
        "base_commit": state["base_commit"],
        "used_for_training": bool(state["used_for_training"]),
        "baseline_observation_sha256": hashlib.sha256(baseline.encode("utf-8")).hexdigest(),
        "prefix_action_count": prefix_count,
        "prefix_actions": prefix_actions,
        "prompt": [*initial, *prefix_messages],
    }


def _session_steps(path: Path, task_id: str, session_id: str) -> list[dict[str, Any]]:
    sessions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _jsonl(path):
        request_path = str(row.get("path", ""))
        if row.get("request", {}).get("action") and request_path.endswith("/actions"):
            parsed = request_path.split("/sessions/", 1)[1].split("/", 1)[0]
            sessions[(str(row.get("task_id")), parsed)].append(row)
    return sessions.get((task_id, session_id), [])


def _execute(environment: RemoteGRPOCodingEnvironment, raw: dict[str, Any]) -> str:
    action = AgentAction.from_dict(raw)
    return getattr(environment, action.kind.value)(**action.arguments)


def _test_status(observation: str) -> tuple[str, tuple[str, ...]]:
    import re

    result = "passed" if observation.startswith("Tests passed") else "failed"
    ids = tuple(sorted(re.findall(r"(?m)^(?:FAILED|PASSED)\s+([^\s]+)", observation)))
    return result, ids


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if __name__ == "__main__":
    main()
