from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .contracts import Trajectory
from .rollout import read_trajectories, write_report


FAILURE_CATEGORIES = (
    "success",
    "context_window_exceeded",
    "protected_test_edit",
    "invalid_action",
    "verifier_timeout",
    "policy_transport_error",
    "policy_protocol_error",
    "patch_failed_verifier",
    "edit_action_failed",
    "no_patch",
)


def classify_trajectory(trajectory: Trajectory) -> str:
    if trajectory.reward.task_success:
        return "success"

    errors = tuple(
        str(error)
        for step in trajectory.steps
        for error in step.policy_metadata.get("errors", ())
    )
    if any(
        marker in error.casefold()
        for error in errors
        for marker in ("maximum context length", "context length", "max_tokens")
    ):
        return "context_window_exceeded"

    observations = "\n".join(step.observation for step in trajectory.steps).casefold()
    if "cannot modify verifier-owned test file" in observations:
        return "protected_test_edit"
    if any(violation.startswith("invalid_action:") for violation in trajectory.reward.violations):
        return "invalid_action"
    if any(
        step.test_result is not None and step.test_result.timed_out
        for step in trajectory.steps
    ):
        return "verifier_timeout"
    if "policy_transport_error" in trajectory.reward.violations:
        return "policy_transport_error"
    if "policy_protocol_error" in trajectory.reward.violations:
        return "policy_protocol_error"
    if trajectory.reward.patch_created or trajectory.changed_files:
        return "patch_failed_verifier"
    if any(
        step.action.kind.value == "replace_text"
        and step.observation.startswith("Tool error:")
        for step in trajectory.steps
    ):
        return "edit_action_failed"
    return "no_patch"


def build_failure_report(trajectories: Iterable[Trajectory]) -> dict[str, Any]:
    items = tuple(trajectories)
    category_counts: Counter[str] = Counter()
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    summaries: list[dict[str, Any]] = []
    for trajectory in items:
        category = classify_trajectory(trajectory)
        category_counts[category] += 1
        by_task[trajectory.task_id][category] += 1
        summaries.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "task_id": trajectory.task_id,
                "repetition": trajectory.repetition,
                "seed": trajectory.seed,
                "category": category,
                "steps": trajectory.reward.steps,
                "tool_calls": trajectory.reward.tool_calls,
                "changed_file_count": len(trajectory.changed_files),
                "violations": list(trajectory.reward.violations),
            }
        )
    return {
        "schema_version": 1,
        "report_type": "trajectory_failure_taxonomy",
        "trial_count": len(items),
        "success_count": category_counts["success"],
        "failure_count": len(items) - category_counts["success"],
        "category_counts": {
            category: category_counts[category]
            for category in FAILURE_CATEGORIES
            if category_counts[category]
        },
        "by_task": {
            task_id: dict(sorted(counts.items()))
            for task_id, counts in sorted(by_task.items())
        },
        "trajectories": summaries,
        "contains_raw_model_or_repository_content": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize trajectory failure categories")
    parser.add_argument("--input", required=True, help="Trajectory JSONL file")
    parser.add_argument("--output", required=True, help="Failure report JSON file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / args.input
    output_path = project_root / args.output
    try:
        trajectories = read_trajectories(input_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    report = build_failure_report(trajectories)
    write_report(report, output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
