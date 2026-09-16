from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def summarize_navigation(
    gold_examples: Iterable[dict[str, Any]],
    traces: Iterable[dict[str, Any]],
    *,
    max_actions: int = 8,
) -> dict[str, Any]:
    paths_by_task: dict[str, set[str]] = defaultdict(set)
    for row in gold_examples:
        if row["stage"] == "inspect":
            paths_by_task[row["task_id"]].add(row["target_action"]["arguments"]["path"])
        elif row["stage"] == "navigation":
            paths_by_task[row["task_id"]].add(row["source_path"])

    sessions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        action = trace.get("request", {}).get("action")
        if not action or "/sessions/" not in trace.get("path", ""):
            continue
        session_id = trace["path"].split("/sessions/", 1)[1].split("/", 1)[0]
        sessions[(trace["task_id"], session_id)].append(action)

    trials = []
    for (task_id, session_id), actions in sessions.items():
        gold_paths = paths_by_task[task_id]
        first_hit = next(
            (
                index
                for index, action in enumerate(actions[:max_actions], 1)
                if action["kind"] == "read_file" and action["arguments"].get("path") in gold_paths
            ),
            None,
        )
        bounded = actions[:max_actions]
        trials.append({
            "task_id": task_id,
            "session_id": session_id,
            "first_hit_step": first_hit,
            "read_paths": [
                action["arguments"].get("path")
                for action in bounded if action["kind"] == "read_file"
            ],
            "search_queries": [
                action["arguments"].get("query")
                for action in bounded if action["kind"] == "search_text"
            ],
        })

    by_task: dict[str, dict[str, Any]] = {}
    for task_id in sorted({trial["task_id"] for trial in trials}):
        task_trials = [trial for trial in trials if trial["task_id"] == task_id]
        hits = sum(trial["first_hit_step"] is not None for trial in task_trials)
        by_task[task_id] = {
            "trial_count": len(task_trials),
            "target_file_hits": hits,
            "target_file_hit_rate": hits / len(task_trials),
        }
    hits = sum(trial["first_hit_step"] is not None for trial in trials)
    return {
        "schema_version": 2,
        "max_actions": max_actions,
        "trial_count": len(trials),
        "target_file_hits": hits,
        "target_file_hit_rate": hits / len(trials) if trials else 0.0,
        "tasks": by_task,
        "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure gold source-file hits in tool traces")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = summarize_navigation(
        _jsonl(Path(args.gold)),
        (row for trace in args.trace for row in _jsonl(Path(trace))),
        max_actions=args.max_actions,
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    main()
