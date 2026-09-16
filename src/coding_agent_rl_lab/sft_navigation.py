from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .sft_grpo import GRPO_ACTION_PROTOCOL, GRPO_SFT_PROMPT_VERSION, GRPO_SFT_SCHEMA


def build_navigation_examples(
    gold_examples: Iterable[dict[str, Any]],
    traces: Iterable[dict[str, Any]],
    *,
    max_gold_paths: int = 3,
    max_prefix_actions: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold = list(gold_examples)
    paths_by_task: dict[str, set[str]] = defaultdict(set)
    for row in gold:
        if row["stage"] == "inspect":
            paths_by_task[row["task_id"]].add(row["target_action"]["arguments"]["path"])
    selected = {task for task, paths in paths_by_task.items() if len(paths) <= max_gold_paths}

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    inspect_by_task_path: dict[tuple[str, str], dict[str, Any]] = {}
    for row in gold:
        if row["task_id"] not in selected or row["stage"] != "inspect":
            continue
        path = row["target_action"]["arguments"]["path"]
        inspect_by_task_path.setdefault((row["task_id"], path), row)
        _append(output, seen, row, row["messages"], "gold-navigation")

    sessions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        if trace.get("task_id") not in selected or "/sessions/" not in trace.get("path", ""):
            continue
        session_id = trace["path"].split("/sessions/", 1)[1].split("/", 1)[0]
        if trace.get("request", {}).get("action"):
            sessions[(trace["task_id"], session_id)].append(trace)

    for (task_id, _), steps in sessions.items():
        seed_row = next(row for row in gold if row["task_id"] == task_id)
        history = [dict(seed_row["messages"][0]), dict(seed_row["messages"][1])]
        for index, step in enumerate(steps[:max_prefix_actions]):
            action = step["request"]["action"]
            observation = step["response"].get("observation", "")
            call = {"name": action["kind"], "arguments": action["arguments"]}
            for path in sorted(paths_by_task[task_id]):
                if _exposes_import(observation, path):
                    target = _retarget(inspect_by_task_path[(task_id, path)], action)
                    _append(
                        output, seen, target, history + [{"role": "assistant", "content": _compact(call)}],
                        "trace-bridge", index,
                    )
            history.extend(
                [
                    {"role": "assistant", "content": _compact(call)},
                    {"role": "tool", "name": action["kind"], "content": observation},
                ]
            )
            for path in sorted(paths_by_task[task_id]):
                if not _exposes_path(observation, path):
                    continue
                target = inspect_by_task_path[(task_id, path)]
                messages = history + [dict(target["messages"][-1])]
                _append(output, seen, target, messages, "trace-navigation", index + 1)
            if action["kind"] in {"replace_lines", "run_verifier", "finish"}:
                break

    stages = defaultdict(int)
    tasks = defaultdict(int)
    for row in output:
        stages[row["navigation_source"]] += 1
        tasks[row["task_id"]] += 1
    report = {
        "schema_version": 1,
        "dataset_schema": GRPO_SFT_SCHEMA,
        "task_set": "train",
        "answer_source": "official_swe_gym_gold_patch",
        "contains_answers": True,
        "prompt_version": GRPO_SFT_PROMPT_VERSION,
        "action_protocol": GRPO_ACTION_PROTOCOL,
        "training_performed": False,
        "trajectory_layout": "cross-task-navigation-prefixes",
        "task_ids": sorted(selected),
        "task_count": len(selected),
        "example_count": len(output),
        "source_counts": dict(sorted(stages.items())),
        "task_example_counts": dict(sorted(tasks.items())),
        "max_gold_paths": max_gold_paths,
        "max_prefix_actions": max_prefix_actions,
    }
    return output, report


def _exposes_path(observation: str, path: str) -> bool:
    return _exposes_import(observation, path) or any(
        marker in observation
        for marker in (f"PATH_MATCH:{path}", f"SUGGESTED_PATH:{path}", f"{path}:")
    )


def _exposes_import(observation: str, path: str) -> bool:
    module = path.removesuffix(".py").replace("/", ".")
    return f"from {module} import " in observation or f"import {module}" in observation


def _retarget(source: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    row = dict(source)
    row["target_action"] = {"kind": action["kind"], "arguments": action["arguments"]}
    row["target_tool_call"] = {"name": action["kind"], "arguments": action["arguments"]}
    return row


def _append(
    output: list[dict[str, Any]],
    seen: set[str],
    source: dict[str, Any],
    messages: list[dict[str, Any]],
    navigation_source: str,
    prefix_actions: int = 0,
) -> None:
    identity = _compact(messages)
    if identity in seen:
        return
    seen.add(identity)
    row = dict(source)
    row["example_id"] = "navigation-sft-" + hashlib.sha256(identity.encode()).hexdigest()[:20]
    row["stage"] = "navigation"
    row["navigation_source"] = navigation_source
    row["prefix_actions"] = prefix_actions
    row["messages"] = messages
    output.append(row)


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cross-task file-navigation SFT examples")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--max-gold-paths", type=int, default=3)
    parser.add_argument("--max-prefix-actions", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    examples, report = build_navigation_examples(
        _jsonl(Path(args.gold)),
        (row for trace in args.trace for row in _jsonl(Path(trace))),
        max_gold_paths=args.max_gold_paths,
        max_prefix_actions=args.max_prefix_actions,
    )
    Path(args.output).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples), encoding="utf-8"
    )
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    main()
