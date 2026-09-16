from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def build_cumulative_examples(
    examples: Iterable[dict[str, Any]],
    report: dict[str, Any],
    *,
    task_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    verify_rows = [
        row
        for row in examples
        if row["stage"] == "verify" and (task_ids is None or row["task_id"] in task_ids)
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in verify_rows:
        grouped.setdefault(row["task_id"], []).append(row)

    cumulative: list[dict[str, Any]] = []
    for task_id, rows in grouped.items():
        rows.sort(key=lambda row: row["hunk_index"])
        history = [dict(message) for message in rows[0]["messages"][:2]]
        seen_calls: set[str] = set()
        sequence = 0
        for row in rows:
            messages = row["messages"]
            for assistant, tool in zip(messages[2:-1:2], messages[3:-1:2], strict=True):
                if assistant["content"] in seen_calls:
                    continue
                cumulative.append(_example(row, history, assistant, sequence))
                history.extend((dict(assistant), dict(tool)))
                seen_calls.add(assistant["content"])
                sequence += 1
        cumulative.append(_example(rows[-1], history, rows[-1]["messages"][-1], sequence))

    selected_ids = list(grouped)
    output_report = {
        **report,
        "task_ids": selected_ids,
        "task_count": len(selected_ids),
        "example_count": len(cumulative),
        "source_example_count": report["example_count"],
        "source_stage_counts": report.get("stage_counts"),
        "stage_counts": {"cumulative": len(cumulative)},
        "source_skipped_hunk_counts": report.get("skipped_hunk_counts"),
        "skipped_hunk_counts": {},
        "trajectory_layout": "cumulative-hunks-single-final-verifier",
        "training_performed": False,
    }
    return cumulative, output_report


def _example(
    template: dict[str, Any],
    history: list[dict[str, Any]],
    assistant: dict[str, Any],
    sequence: int,
) -> dict[str, Any]:
    call = json.loads(assistant["content"])
    target_action = {"kind": call["name"], "arguments": call["arguments"]}
    identity = f"{template['task_id']}\0{sequence}\0{assistant['content']}"
    return {
        **template,
        "example_id": "cumulative-sft-"
        + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
        "stage": "cumulative",
        "messages": [*history, dict(assistant)],
        "target_action": target_action,
        "target_tool_call": call,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine per-hunk GRPO SFT rows into cumulative tool trajectories")
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--task-id", action="append", default=[])
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_report = json.loads(Path(args.input_report).read_text(encoding="utf-8"))
    cumulative, report = build_cumulative_examples(
        rows, source_report, task_ids=set(args.task_id) or None
    )
    Path(args.output).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cumulative),
        encoding="utf-8",
    )
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
