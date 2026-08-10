from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import AgentAction, CodingTask, DatasetSplit


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

