from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .contracts import ActionKind, Trajectory
from .rollout import read_trajectories, write_report


FAILURE_CATEGORIES = (
    "success",
    "infra_error",
    "context_window_exceeded",
    "protected_test_edit",
    "invalid_action",
    "verifier_timeout",
    "policy_transport_error",
    "policy_protocol_error",
    "loop",
    "patch_failed_verifier",
    "edit_action_failed",
    "no_edit_attempt",
    "no_patch",
)

INFRA_ERROR_CATEGORIES = (
    "infra_error",
    "context_window_exceeded",
    "policy_transport_error",
)

_CONTEXT_MARKERS = ("maximum context length", "context length", "max_tokens")
_INFRA_MARKERS = (
    "connection refused",
    "connection reset",
    "server disconnected",
    "temporary failure in name resolution",
    "timed out",
    "http 400",
    "http 404",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
)
_LOOP_MARKERS = (
    "repeated rejection",
    "do not repeat the same action",
    "do not repeat a search_text query",
    "do not repeat list_files",
    "do not reread an unchanged file",
    "finish refused",
)
_EDIT_KINDS = {ActionKind.REPLACE_TEXT, ActionKind.REPLACE_LINES}


def classify_trajectory(trajectory: Trajectory) -> str:
    """Attribute one failed trial, separating harness faults from policy faults.

    Context overflow and transport failures say nothing about the policy, so they are
    classified before any behaviour-based category. `loop` and `no_edit_attempt` split
    the old `no_patch` catch-all, which used to hide that most weak-policy trials never
    reached an edit at all.
    """

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
        for marker in _CONTEXT_MARKERS
    ):
        return "context_window_exceeded"
    if "policy_transport_error" in trajectory.reward.violations or any(
        marker in error.casefold() for error in errors for marker in _INFRA_MARKERS
    ):
        return "infra_error"

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
    if "policy_protocol_error" in trajectory.reward.violations:
        return "policy_protocol_error"
    if trajectory.reward.patch_created or trajectory.changed_files:
        return "patch_failed_verifier"
    edit_attempts = tuple(
        step for step in trajectory.steps if step.action.kind in _EDIT_KINDS
    )
    if any(step.observation.startswith("Tool error:") for step in edit_attempts):
        return "edit_action_failed"
    if not edit_attempts:
        if loop_rejection_count(trajectory) >= _loop_threshold(trajectory):
            return "loop"
        return "no_edit_attempt"
    return "no_patch"


def loop_rejection_count(trajectory: Trajectory) -> int:
    """Count productive-actions refusals, falling back to observations for old files."""

    observed = sum(
        1
        for step in trajectory.steps
        if step.observation.startswith("Tool error:")
        and any(marker in step.observation.casefold() for marker in _LOOP_MARKERS)
    )
    return max(trajectory.reward.loop_rejections, observed)


def _loop_threshold(trajectory: Trajectory) -> int:
    """A trial is `loop` when a third of its steps were refusals, and at least three."""

    return max(3, (trajectory.reward.steps + 2) // 3)


def build_failure_report(trajectories: Iterable[Trajectory]) -> dict[str, Any]:
    items = tuple(trajectories)
    category_counts: Counter[str] = Counter()
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    summaries: list[dict[str, Any]] = []
    declared_trials = 0
    resolved_trials = 0
    regressed_trials = 0
    collection_error_trials = 0
    for trajectory in items:
        category = classify_trajectory(trajectory)
        category_counts[category] += 1
        by_task[trajectory.task_id][category] += 1
        verifier = _verifier_summary(trajectory)
        declared_trials += bool(verifier["graded_targets_declared"])
        resolved_trials += bool(verifier["fail_to_pass_resolved"])
        regressed_trials += bool(verifier["pass_to_pass_regressed"])
        collection_error_trials += bool(verifier["collection_error"])
        summaries.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "task_id": trajectory.task_id,
                "repetition": trajectory.repetition,
                "seed": trajectory.seed,
                "category": category,
                "steps": trajectory.reward.steps,
                "tool_calls": trajectory.reward.tool_calls,
                "loop_rejections": loop_rejection_count(trajectory),
                "infra_error": category in INFRA_ERROR_CATEGORIES,
                "changed_file_count": len(trajectory.changed_files),
                "violations": list(trajectory.reward.violations),
                **verifier,
            }
        )
    infra_failures = sum(category_counts[name] for name in INFRA_ERROR_CATEGORIES)
    return {
        "schema_version": 1,
        "report_type": "trajectory_failure_taxonomy",
        "trial_count": len(items),
        "success_count": category_counts["success"],
        "failure_count": len(items) - category_counts["success"],
        "infra_error_count": infra_failures,
        "policy_failure_count": len(items) - category_counts["success"] - infra_failures,
        "category_counts": {
            category: category_counts[category]
            for category in FAILURE_CATEGORIES
            if category_counts[category]
        },
        "by_task": {
            task_id: dict(sorted(counts.items()))
            for task_id, counts in sorted(by_task.items())
        },
        "graded_failure_attribution": {
            "trials_with_declared_targets": declared_trials,
            "trials_with_any_fail_to_pass_resolved": resolved_trials,
            "trials_with_pass_to_pass_regression": regressed_trials,
            "trials_with_collection_error": collection_error_trials,
        },
        "trajectories": summaries,
        "contains_raw_model_or_repository_content": False,
    }


def _verifier_summary(trajectory: Trajectory) -> dict[str, Any]:
    """Node-level verifier attribution, with explicit unknowns instead of fake zeros."""

    verifier = trajectory.verifier
    if verifier is None:
        return {
            "graded_targets_declared": False,
            "fail_to_pass_total": 0,
            "fail_to_pass_resolved": None,
            "pass_to_pass_total": 0,
            "pass_to_pass_regressed": None,
            "collection_error": False,
        }
    return {
        "graded_targets_declared": verifier.node_targets_declared,
        "fail_to_pass_total": verifier.fail_to_pass_total,
        "fail_to_pass_resolved": verifier.fail_to_pass_resolved,
        "pass_to_pass_total": verifier.pass_to_pass_total,
        "pass_to_pass_regressed": verifier.pass_to_pass_regressed,
        "collection_error": verifier.collection_error,
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
