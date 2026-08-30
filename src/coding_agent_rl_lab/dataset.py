from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import AgentAction, CodingTask, DatasetSplit, Trajectory


class DatasetError(ValueError):
    pass


def load_tasks(path: str | Path) -> tuple[CodingTask, ...]:
    rows = _load_jsonl(path)
    tasks: list[CodingTask] = []
    seen: set[str] = set()
    for line_number, row in rows:
        try:
            task = CodingTask(
                task_id=str(row["task_id"]),
                issue=str(row["issue"]),
                fixture_path=str(row["fixture_path"]),
                base_commit=str(row["base_commit"]),
                test_command=tuple(str(item) for item in row["test_command"]),
                split=DatasetSplit(row["split"]),
                provenance=str(row["provenance"]),
                max_steps=int(row.get("max_steps", 8)),
                metadata=dict(row.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetError(f"invalid task at line {line_number}: {exc}") from exc
        if task.task_id in seen:
            raise DatasetError(f"duplicate task_id at line {line_number}: {task.task_id}")
        seen.add(task.task_id)
        tasks.append(task)
    if not tasks:
        raise DatasetError("task dataset must not be empty")
    return tuple(tasks)


def load_reference_actions(path: str | Path) -> dict[str, tuple[AgentAction, ...]]:
    rows = _load_jsonl(path)
    result: dict[str, tuple[AgentAction, ...]] = {}
    for line_number, row in rows:
        try:
            task_id = str(row["task_id"])
            actions = tuple(AgentAction.from_dict(item) for item in row["actions"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetError(f"invalid reference actions at line {line_number}: {exc}") from exc
        if task_id in result:
            raise DatasetError(f"duplicate reference task_id at line {line_number}: {task_id}")
        if not actions:
            raise DatasetError(f"reference actions must not be empty for {task_id}")
        result[task_id] = actions
    return result


def build_trajectory_dataset_audit(
    tasks: tuple[CodingTask, ...],
    trajectories: tuple[Trajectory, ...],
) -> dict[str, Any]:
    """Summarize data eligibility and fail closed on split/provenance mismatches."""

    task_by_id = {task.task_id: task for task in tasks}
    integrity_errors: list[str] = []
    seen_trajectory_ids: set[str] = set()
    content_fingerprints: dict[str, int] = {}
    split_counts = {split.value: 0 for split in DatasetSplit}
    outcome_counts = {"success": 0, "failure": 0}
    failure_taxonomy: dict[str, int] = {}
    training_eligible = 0
    answer_bearing = 0

    snapshot_splits: dict[tuple[str, str, str], set[str]] = {}
    for task in tasks:
        split_counts[task.split.value] += 1
        snapshot = (task.provenance, task.fixture_path, task.base_commit)
        snapshot_splits.setdefault(snapshot, set()).add(task.split.value)

    split_leakage = [
        {
            "provenance": snapshot[0],
            "fixture_path": snapshot[1],
            "base_commit": snapshot[2],
            "splits": sorted(splits),
        }
        for snapshot, splits in sorted(snapshot_splits.items())
        if len(splits) > 1
    ]
    if split_leakage:
        integrity_errors.append("repository snapshots occur in more than one dataset split")

    for trajectory in trajectories:
        if trajectory.trajectory_id in seen_trajectory_ids:
            integrity_errors.append(f"duplicate trajectory_id: {trajectory.trajectory_id}")
        seen_trajectory_ids.add(trajectory.trajectory_id)
        task = task_by_id.get(trajectory.task_id)
        if task is None:
            integrity_errors.append(f"unknown task_id: {trajectory.task_id}")
        elif (
            trajectory.task_split is not task.split
            or trajectory.task_provenance != task.provenance
            or trajectory.task_base_commit != task.base_commit
        ):
            integrity_errors.append(f"task provenance mismatch: {trajectory.trajectory_id}")

        payload = trajectory.to_dict()
        for volatile in ("trajectory_id", "repetition", "seed"):
            payload.pop(volatile, None)
        for step in payload["steps"]:
            if step["test_result"] is not None:
                step["test_result"].pop("duration_ms", None)
        fingerprint = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        content_fingerprints[fingerprint] = content_fingerprints.get(fingerprint, 0) + 1

        if trajectory.reward.task_success:
            outcome_counts["success"] += 1
        else:
            outcome_counts["failure"] += 1
            category = _failure_category(trajectory)
            failure_taxonomy[category] = failure_taxonomy.get(category, 0) + 1

        contains_answers = trajectory.policy.metadata.get("contains_answers") is True
        answer_bearing += int(contains_answers)
        if trajectory.reward.task_success and not trajectory.reward.violations and not contains_answers:
            training_eligible += 1

    duplicate_groups = sum(count > 1 for count in content_fingerprints.values())
    duplicate_records = sum(count - 1 for count in content_fingerprints.values() if count > 1)
    return {
        "schema_version": 1,
        "integrity_ok": not integrity_errors,
        "integrity_errors": integrity_errors,
        "task_split_counts": split_counts,
        "trajectory_count": len(trajectories),
        "outcomes": outcome_counts,
        "failure_taxonomy": dict(sorted(failure_taxonomy.items())),
        "answer_bearing_count": answer_bearing,
        "training_eligible_count": training_eligible,
        "duplicate_content_groups": duplicate_groups,
        "duplicate_content_records": duplicate_records,
        "split_leakage": split_leakage,
    }


def _failure_category(trajectory: Trajectory) -> str:
    if trajectory.reward.violations:
        return "policy_or_safety_violation"
    if trajectory.reward.patch_created:
        return "tests_failed_with_patch"
    return "tests_failed_without_patch"


def _load_jsonl(path: str | Path) -> list[tuple[int, dict[str, Any]]]:
    target = Path(path)
    rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"invalid JSON at line {line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise DatasetError(f"line {line_number} must contain a JSON object")
        rows.append((line_number, value))
    return rows
