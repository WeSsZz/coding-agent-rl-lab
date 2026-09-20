from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .contracts import ActionKind, AgentAction, CodingTask, DatasetSplit, TrajectoryStep
from .environment import render_numbered_window
from .model_policy import PROMPT_VERSION, build_action_messages
from .swe_gym_smoke import _download_pinned_rows, pinned_rows_for_task_set


class SFTDatasetError(ValueError):
    pass


@dataclass(frozen=True)
class PatchHunk:
    old_path: str
    new_path: str
    old_start: int
    old_text: str
    new_text: str


_HUNK_HEADER = re.compile(r"^@@ -(?P<old_start>\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")
MAX_TARGET_ACTION_CHARS = 4096


def parse_unified_diff(patch: str) -> tuple[PatchHunk, ...]:
    lines = patch.splitlines(keepends=True)
    hunks: list[PatchHunk] = []
    old_path: str | None = None
    new_path: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("--- "):
            old_path = _patch_header_path(line[4:])
            index += 1
            if index >= len(lines) or not lines[index].startswith("+++ "):
                raise SFTDatasetError("unified diff --- header is not followed by +++")
            new_path = _patch_header_path(lines[index][4:])
            index += 1
            continue
        match = _HUNK_HEADER.match(line)
        if match:
            if old_path is None or new_path is None:
                raise SFTDatasetError("unified diff hunk appears before file headers")
            old_lines: list[str] = []
            new_lines: list[str] = []
            previous_marker: str | None = None
            index += 1
            while index < len(lines):
                hunk_line = lines[index]
                if hunk_line.startswith(("diff --git ", "--- ", "@@ ")):
                    break
                if hunk_line.startswith("\\ No newline at end of file"):
                    if previous_marker in {" ", "-"} and old_lines:
                        old_lines[-1] = old_lines[-1].rstrip("\r\n")
                    if previous_marker in {" ", "+"} and new_lines:
                        new_lines[-1] = new_lines[-1].rstrip("\r\n")
                    index += 1
                    continue
                if not hunk_line or hunk_line[0] not in {" ", "+", "-"}:
                    break
                previous_marker = hunk_line[0]
                content = hunk_line[1:]
                if previous_marker in {" ", "-"}:
                    old_lines.append(content)
                if previous_marker in {" ", "+"}:
                    new_lines.append(content)
                index += 1
            hunks.append(
                PatchHunk(
                    old_path=old_path,
                    new_path=new_path,
                    old_start=int(match.group("old_start")),
                    old_text="".join(old_lines),
                    new_text="".join(new_lines),
                )
            )
            continue
        index += 1
    return tuple(hunks)


def build_train_gold_sft_dataset(
    rows: Iterable[dict[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    items = tuple(rows)
    if not items:
        raise SFTDatasetError("at least one train row is required")
    allowed = {item.instance_id: item for item in pinned_rows_for_task_set("train")}
    seen: set[str] = set()
    examples: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    example_counts: Counter[str] = Counter()
    task_ids: list[str] = []

    for row in items:
        task_id = row.get("instance_id")
        if not isinstance(task_id, str) or task_id not in allowed:
            raise SFTDatasetError(f"row is outside the pinned train split: {task_id!r}")
        if task_id in seen:
            raise SFTDatasetError(f"duplicate train row: {task_id}")
        seen.add(task_id)
        pinned = allowed[task_id]
        if row.get("base_commit") != pinned.base_commit:
            raise SFTDatasetError(f"base commit mismatch for {task_id}")
        if row.get("repo") != "getmoto/moto" or row.get("version") != "5.0":
            raise SFTDatasetError(f"repository metadata mismatch for {task_id}")
        issue = row.get("problem_statement")
        patch = row.get("patch")
        if not isinstance(issue, str) or not issue.strip():
            raise SFTDatasetError(f"train row {task_id} has no problem statement")
        if not isinstance(patch, str) or not patch.strip():
            raise SFTDatasetError(f"train row {task_id} has no gold patch")
        task = CodingTask(
            task_id=task_id,
            issue=issue,
            fixture_path=None,
            base_commit=pinned.base_commit,
            test_command=("verifier",),
            split=DatasetSplit.DEVELOPMENT,
            provenance="swe-gym:train-gold-supervision:v1",
            max_steps=12,
            metadata={"repo": "getmoto/moto", "version": "5.0"},
        )
        initial_observation = _training_initial_observation(row)
        usable_hunks = 0
        for hunk_index, hunk in enumerate(parse_unified_diff(patch), start=1):
            reason = _unsupported_hunk_reason(hunk)
            if reason is not None:
                skipped[reason] += 1
                continue
            usable_hunks += 1
            for example in _hunk_examples(
                task,
                hunk,
                hunk_index=hunk_index,
                initial_observation=initial_observation,
            ):
                examples.append(example)
                example_counts[example["stage"]] += 1
        if usable_hunks == 0:
            raise SFTDatasetError(f"train row {task_id} has no supported source-edit hunks")
        task_ids.append(task_id)

    report = {
        "schema_version": 1,
        "dataset_schema": "coding-agent-gold-sft-v1",
        "task_set": "train",
        "task_ids": task_ids,
        "task_count": len(task_ids),
        "example_count": len(examples),
        "stage_counts": dict(sorted(example_counts.items())),
        "skipped_hunk_counts": dict(sorted(skipped.items())),
        "contains_answers": True,
        "answer_source": "official_swe_gym_gold_patch",
        "prompt_version": PROMPT_VERSION,
        "max_target_action_chars": MAX_TARGET_ACTION_CHARS,
        "training_performed": False,
        "intended_use": "train-split-only-supervised-tool-warm-start",
    }
    return tuple(examples), report


def write_sft_dataset(
    examples: tuple[dict[str, Any], ...],
    report: dict[str, Any],
    *,
    output_path: str | Path,
    report_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(example, ensure_ascii=False) + "\n" for example in examples),
        encoding="utf-8",
    )
    report_target = Path(report_path)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an explicitly answer-containing SFT warm-start dataset from train-only gold patches"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="JSONL containing full official train rows with gold patches")
    source.add_argument(
        "--download-pinned-train",
        action="store_true",
        help="Download only the six pinned train rows in memory; do not write a raw row cache.",
    )
    parser.add_argument(
        "--output",
        default="work/private/swe-gym-train-gold-sft-v1.jsonl",
    )
    parser.add_argument(
        "--report",
        default="work/private/swe-gym-train-gold-sft-v1-report.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.download_pinned_train:
        rows = _download_pinned_rows(pinned_rows_for_task_set("train"))
    else:
        rows = _load_rows(project_root / args.input)
    examples, report = build_train_gold_sft_dataset(rows)
    write_sft_dataset(
        examples,
        report,
        output_path=project_root / args.output,
        report_path=project_root / args.report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _load_rows(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SFTDatasetError(f"invalid JSON on row {line_number}") from exc
        if not isinstance(row, dict):
            raise SFTDatasetError(f"row {line_number} must be an object")
        rows.append(row)
    return tuple(rows)


def _hunk_examples(
    task: CodingTask,
    hunk: PatchHunk,
    *,
    hunk_index: int,
    initial_observation: str,
) -> tuple[dict[str, Any], ...]:
    path = hunk.new_path
    query = path
    search_action = AgentAction(ActionKind.SEARCH_TEXT, {"query": query})
    source_line = next(
        (line.strip() for line in hunk.old_text.splitlines() if line.strip()),
        PurePosixPath(path).name,
    )
    search_observation = f"{path}:{hunk.old_start}:{source_line[:300]}"
    range_start = max(1, hunk.old_start - 10)
    range_end = max(range_start + 19, hunk.old_start + len(hunk.old_text.splitlines()) + 9)
    range_end = min(range_start + 399, range_end)
    read_action = AgentAction(
        ActionKind.READ_FILE,
        {"path": path, "start_line": range_start, "end_line": range_end},
    )
    replace_action = AgentAction(
        ActionKind.REPLACE_TEXT,
        {"path": path, "old": hunk.old_text, "new": hunk.new_text},
    )
    run_tests_action = AgentAction(ActionKind.RUN_TESTS)

    search_step = TrajectoryStep(1, search_action, search_observation, False)
    # The observation has to be rendered exactly the way the live environment renders it, line
    # numbers and closing footer included. A `read_file` observation is the only place the policy
    # can see the numbering, so an `edit` target whose history shows bare unnumbered source is a
    # target the model can copy verbatim - which is precisely the `old` a numbered observation
    # must never become, and what a warm start trained this way does at inference time.
    read_observation = render_numbered_window(
        hunk.old_text,
        None,
        max_lines=max(range_end - range_start + 1, len(hunk.old_text.splitlines())),
    )
    read_step = TrajectoryStep(2, read_action, read_observation, False)
    replace_step = TrajectoryStep(3, replace_action, f"Updated {path}.", False)
    stages = (
        ("locate", (), search_action),
        ("inspect", (search_step,), read_action),
        ("edit", (search_step, read_step), replace_action),
        ("verify", (search_step, read_step, replace_step), run_tests_action),
    )
    return tuple(
        _sft_example(
            task,
            history,
            target,
            stage=stage,
            hunk_index=hunk_index,
            path=path,
            initial_observation=initial_observation,
        )
        for stage, history, target in stages
    )


def _sft_example(
    task: CodingTask,
    history: tuple[TrajectoryStep, ...],
    target: AgentAction,
    *,
    stage: str,
    hunk_index: int,
    path: str,
    initial_observation: str,
) -> dict[str, Any]:
    target_text = json.dumps(target.to_dict(), ensure_ascii=False, separators=(",", ":"))
    identity = f"{task.task_id}\0{path}\0{hunk_index}\0{stage}\0{target_text}"
    messages = [
        *build_action_messages(task, history, initial_observation),
        {"role": "assistant", "content": target_text},
    ]
    return {
        "schema_version": 1,
        "example_id": "sft-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
        "task_id": task.task_id,
        "task_set": "train",
        "stage": stage,
        "source_path": path,
        "hunk_index": hunk_index,
        "prompt_version": PROMPT_VERSION,
        "messages": messages,
        "target_action": target.to_dict(),
        "contains_answers": True,
        "answer_source": "official_swe_gym_gold_patch",
    }


def _unsupported_hunk_reason(hunk: PatchHunk) -> str | None:
    if hunk.old_path == "/dev/null" or hunk.new_path == "/dev/null":
        return "file_creation_or_deletion"
    if hunk.old_path != hunk.new_path:
        return "file_rename"
    path = PurePosixPath(hunk.new_path)
    if path.is_absolute() or ".." in path.parts:
        return "unsafe_path"
    folded_parts = {part.casefold() for part in path.parts}
    filename = path.name.casefold()
    if (
        "tests" in folded_parts
        or "test" in folded_parts
        or filename.startswith("test_")
        or filename.endswith("_test.py")
    ):
        return "test_file"
    if not hunk.old_text:
        return "empty_old_text"
    if hunk.old_text == hunk.new_text:
        return "no_op"
    action = AgentAction(
        ActionKind.REPLACE_TEXT,
        {"path": hunk.new_path, "old": hunk.old_text, "new": hunk.new_text},
    )
    target_text = json.dumps(action.to_dict(), ensure_ascii=False, separators=(",", ":"))
    if len(target_text) > MAX_TARGET_ACTION_CHARS:
        return "oversized_target_action"
    return None


def _training_initial_observation(row: dict[str, Any]) -> str:
    raw_tests = row.get("FAIL_TO_PASS", ())
    if isinstance(raw_tests, str):
        try:
            decoded = json.loads(raw_tests)
        except json.JSONDecodeError:
            decoded = ()
        tests = decoded if isinstance(decoded, list) else ()
    elif isinstance(raw_tests, list):
        tests = raw_tests
    else:
        tests = ()
    rendered = "\n".join(str(test) for test in tests[:20])
    suffix = f"\nFailing tests:\n{rendered}" if rendered else ""
    return f"Baseline verifier result:\nTests failed (exit=1).{suffix}"


def _patch_header_path(value: str) -> str:
    path = value.rstrip("\r\n").split("\t", 1)[0]
    if path in {"/dev/null", "dev/null"}:
        return "/dev/null"
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path


if __name__ == "__main__":
    main()
