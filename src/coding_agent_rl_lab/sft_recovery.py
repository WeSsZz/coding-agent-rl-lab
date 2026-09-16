from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def build_suggested_path_examples(
    gold_examples: Iterable[dict[str, Any]],
    traces: Iterable[dict[str, Any]],
    *,
    task_id: str,
    suggested_path: str,
) -> list[dict[str, Any]]:
    target = next(
        row
        for row in gold_examples
        if row["task_id"] == task_id
        and row["target_action"]["kind"] == "read_file"
        and row["target_action"]["arguments"]["path"] == suggested_path
    )
    system, user = target["messages"][:2]
    assistant_target = target["messages"][-1]
    marker = f"SUGGESTED_PATH:{suggested_path}"
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for trace in traces:
        action = trace.get("request", {}).get("action")
        observation = trace.get("response", {}).get("observation", "")
        if (
            trace.get("task_id") != task_id
            or not isinstance(action, dict)
            or action.get("kind") != "search_text"
            or marker not in observation
        ):
            continue
        tool_call = {"name": action["kind"], "arguments": action["arguments"]}
        call_text = json.dumps(tool_call, ensure_ascii=False, separators=(",", ":"))
        if call_text in seen:
            continue
        seen.add(call_text)
        identity = f"{task_id}\0{call_text}\0{suggested_path}"
        output.append(
            {
                **target,
                "example_id": "recovery-sft-"
                + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
                "stage": "recovery",
                "messages": [
                    dict(system),
                    dict(user),
                    {"role": "assistant", "content": call_text},
                    {"role": "tool", "name": "search_text", "content": observation},
                    dict(assistant_target),
                ],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train-only SFT examples that follow a search tool's suggested path"
    )
    parser.add_argument("--gold", required=True)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--suggested-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    gold = _jsonl(Path(args.gold))
    traces = [row for path in args.trace for row in _jsonl(Path(path))]
    examples = build_suggested_path_examples(
        gold, traces, task_id=args.task_id, suggested_path=args.suggested_path
    )
    Path(args.output).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples),
        encoding="utf-8",
    )
    source_report = json.loads(Path(args.gold).with_name(Path(args.gold).stem + "-report.json").read_text())
    report = {
        **source_report,
        "task_ids": [args.task_id],
        "task_count": 1,
        "example_count": len(examples),
        "stage_counts": {"recovery": len(examples)},
        "trajectory_layout": "follow-search-suggested-path",
        "training_performed": False,
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    main()
