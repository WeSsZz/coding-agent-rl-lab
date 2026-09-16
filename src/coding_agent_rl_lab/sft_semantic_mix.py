from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .sft_grpo import GRPO_ACTION_PROTOCOL, GRPO_SFT_PROMPT_VERSION, GRPO_SFT_SCHEMA
from .sft_semantic_recovery import (
    MIXED_AUDITED_ANSWER_SOURCE,
    SEMANTIC_RECOVERY_ANSWER_SOURCE,
)


class SemanticMixError(ValueError):
    pass


def build_semantic_training_mix(
    recovery_rows: Iterable[dict[str, Any]],
    audited_rows: Iterable[dict[str, Any]],
    *,
    recovery_per_state: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if recovery_per_state <= 0:
        raise SemanticMixError("recovery_per_state must be positive")
    recovery = list(recovery_rows)
    audited = list(audited_rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in recovery:
        _validate_row(row, expected_source=SEMANTIC_RECOVERY_ANSWER_SOURCE)
        metadata = row.get("semantic_recovery")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("state_id"), str):
            raise SemanticMixError("recovery row lacks semantic_recovery.state_id")
        grouped[metadata["state_id"]].append(row)
    if len(grouped) < 3:
        raise SemanticMixError("training mix requires at least three recovery states")
    for row in audited:
        _validate_row(row, expected_source="official_swe_gym_gold_patch")

    selected_by_state: dict[str, list[dict[str, Any]]] = {}
    for state_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["semantic_recovery"]["sequence"])
        selected_by_state[state_id] = _select_recovery(rows, recovery_per_state)
    selected_recovery = _round_robin(selected_by_state)
    if len(selected_recovery) % 3:
        raise SemanticMixError("selected recovery count must be divisible by three")
    audited_count = len(selected_recovery) // 3
    selected_audited = _select_audited(audited, audited_count)
    mixed: list[dict[str, Any]] = []
    for index, row in enumerate(selected_recovery, start=1):
        mixed.append(row)
        if index % 3 == 0:
            mixed.append(selected_audited[index // 3 - 1])

    example_ids = [row["example_id"] for row in mixed]
    if len(set(example_ids)) != len(example_ids):
        raise SemanticMixError("mixed dataset contains duplicate example ids")
    recovery_count = len(selected_recovery)
    audit_count = len(selected_audited)
    report = {
        "schema_version": 1,
        "dataset_schema": GRPO_SFT_SCHEMA,
        "task_set": "train",
        "task_ids": sorted({row["task_id"] for row in mixed}),
        "task_count": len({row["task_id"] for row in mixed}),
        "example_count": len(mixed),
        "contains_answers": True,
        "answer_source": MIXED_AUDITED_ANSWER_SOURCE,
        "answer_sources": [
            SEMANTIC_RECOVERY_ANSWER_SOURCE,
            "official_swe_gym_gold_patch",
        ],
        "prompt_version": GRPO_SFT_PROMPT_VERSION,
        "action_protocol": GRPO_ACTION_PROTOCOL,
        "stage_counts": {
            "semantic-recovery": recovery_count,
            "audited-train-tool": audit_count,
        },
        "mixture_percent": {
            "semantic-recovery": 100 * recovery_count // len(mixed),
            "audited-train-tool": 100 * audit_count // len(mixed),
        },
        "state_example_counts": {
            state_id: len(rows) for state_id, rows in selected_by_state.items()
        },
        "selected_recovery_example_ids": [row["example_id"] for row in selected_recovery],
        "selected_audited_example_ids": [row["example_id"] for row in selected_audited],
        "ordering": "three round-robin recovery rows followed by one audited row",
        "training_performed": False,
        "intended_use": "train-split-only-completion-loss-semantic-recovery-mix",
    }
    return mixed, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic 75/25 semantic recovery SFT mix")
    parser.add_argument("--recovery", required=True)
    parser.add_argument("--audited", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--recovery-per-state", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output)
    report_path = Path(args.report)
    for target in (output, report_path):
        if target.exists():
            raise SemanticMixError(f"refusing to overwrite existing output: {target}")
    mixed, report = build_semantic_training_mix(
        _jsonl(Path(args.recovery)),
        _jsonl(Path(args.audited)),
        recovery_per_state=args.recovery_per_state,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in mixed),
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _select_recovery(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise SemanticMixError(f"recovery state has only {len(rows)} rows, needs {count}")
    priority: list[dict[str, Any]] = []
    priority.extend(
        row for row in rows if row["target_action"]["kind"] in {"replace_text", "replace_lines"}
    )
    priority.extend(row for row in rows if row["target_action"]["kind"] == "run_tests")
    priority.extend(
        reversed([row for row in rows if row["target_action"]["kind"] == "read_file"])
    )
    priority.extend(
        reversed([row for row in rows if row["target_action"]["kind"] == "search_text"])
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in priority:
        if row["example_id"] in seen:
            continue
        selected.append(row)
        seen.add(row["example_id"])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise SemanticMixError("could not select enough recovery rows")
    return sorted(selected, key=lambda row: row["semantic_recovery"]["sequence"])


def _select_audited(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = [
        next((row for row in rows if row["target_action"]["kind"] == "search_text"), None),
        next((row for row in rows if row["target_action"]["kind"] == "read_file"), None),
        next(
            (
                row
                for row in reversed(rows)
                if row["target_action"]["kind"] in {"replace_text", "replace_lines"}
            ),
            None,
        ),
        next((row for row in reversed(rows) if row["target_action"]["kind"] == "run_tests"), None),
    ]
    selected = [row for row in ordered if row is not None]
    if len(selected) < count:
        selected.extend(row for row in rows if row not in selected)
    selected = selected[:count]
    if len(selected) != count:
        raise SemanticMixError(f"audited dataset has only {len(selected)} selectable rows, needs {count}")
    return selected


def _round_robin(grouped: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    state_ids = sorted(grouped)
    lengths = {len(grouped[state_id]) for state_id in state_ids}
    if len(lengths) != 1:
        raise SemanticMixError("recovery state selection is not balanced")
    return [
        grouped[state_id][sequence]
        for sequence in range(next(iter(lengths)))
        for state_id in state_ids
    ]


def _validate_row(row: Any, *, expected_source: str) -> None:
    if (
        not isinstance(row, dict)
        or row.get("task_set") != "train"
        or row.get("contains_answers") is not True
        or row.get("answer_source") != expected_source
        or row.get("dataset_schema") is not None
        or not isinstance(row.get("target_action"), dict)
        or not isinstance(row.get("example_id"), str)
    ):
        raise SemanticMixError(f"invalid {expected_source} row")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SemanticMixError(f"empty input: {path}")
    return rows


if __name__ == "__main__":
    main()
