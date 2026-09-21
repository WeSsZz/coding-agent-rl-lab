"""Audit the *first* critical failure of each trial in an A/B/C diagnostic run.

Loop repetitions are noise: a policy that repeats a refused action twenty times produced one mistake,
not twenty. This reads the raw trajectories and reports, per trial:

* the first refused action, the observation the policy was answering, and the refusal text itself
  (the "last instruction" it was given);
* for the first edit refused as unparseable, the action's arguments next to the base-commit code at
  those lines, classified as indentation, range/line-number, or wrong file;
* every applied edit and the tests still failing at the end;
* whether each trial's edits landed in a file the repair touches and whether the target lines had
  been read first - `read_file` gates the file, not the range, so a policy can rewrite lines it has
  never seen and the mistake surfaces as a syntax error instead of a missing-context error.

Nothing here consults the model again; it is all re-derivable from the stored run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

REREAD = "do not reread an unchanged file"


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def shown_lines(observation: str) -> list[int]:
    return [int(match) for match in re.findall(r"^(\d+): ", observation, re.M)]


def read_intervals(trajectory: dict[str, Any]) -> dict[str, list[tuple[int, int]]]:
    """Successful reads per path, as (first, last) intervals, in the order they happened."""

    intervals: dict[str, list[tuple[int, int]]] = {}
    for step in trajectory["steps"]:
        action = step["action"]
        observation = step.get("observation") or ""
        if action["kind"] != "read_file" or observation.startswith("Tool error"):
            continue
        path = action["arguments"].get("path", "")
        first, last = action["arguments"].get("start_line"), action["arguments"].get("end_line")
        if first is None or last is None:
            numbers = shown_lines(observation)
            if not numbers:
                continue
            interval = (min(numbers), max(numbers))
        else:
            interval = (int(first), int(last))
        intervals.setdefault(path, []).append(interval)
    return intervals


def classify_unparseable(
    trajectory: dict[str, Any], step: dict[str, Any], source_root: Path, gold_paths: set[str]
) -> dict[str, Any]:
    """Two independent axes, because one edit usually fails for both reasons at once.

    * ``first_line_unindented`` - the replacement's first line starts at column 0, which is the
      proximate cause of the `IndentationError`;
    * ``target_range`` - whether the policy had been shown the lines it rewrote. `read_file` gates
      the file, not the range, so "never_read" means the indentation had to be guessed.
    """

    arguments = step["action"]["arguments"]
    path = arguments.get("path", "")
    new_text = arguments.get("new") or arguments.get("updated") or ""
    lines_of_new = new_text.splitlines()
    unindented = bool(lines_of_new) and bool(lines_of_new[0]) and not lines_of_new[0][:1].isspace()

    if arguments.get("start_line") is not None:
        target: tuple[int, int] | None = (
            int(arguments["start_line"]),
            int(arguments["end_line"]),
        )
    else:
        first_line = (arguments.get("old") or "").splitlines()[:1]
        cached = source_root / trajectory["task_id"] / path
        if not first_line or not cached.is_file():
            target = None
        else:
            lines = cached.read_text(encoding="utf-8").splitlines()
            index = next(
                (number for number, line in enumerate(lines, 1) if first_line[0] in line), None
            )
            target = (index, index) if index else None

    seen = read_intervals(trajectory).get(path, [])
    if target is None:
        target_state = "unknown"
    elif not seen:
        target_state = "file_never_read"
    elif any(first <= target[0] and target[1] <= last for first, last in seen):
        target_state = "read"
    elif any(first <= target[0] <= last or first <= target[1] <= last for first, last in seen):
        target_state = "partly_read"
    else:
        target_state = "never_read"

    return {
        "first_line_unindented": unindented,
        "target_range": target_state,
        "target_in_a_repair_file": path in gold_paths,
        "target": target,
    }


def audit(
    trajectory: dict[str, Any],
    *,
    gold_paths: set[str],
    source_root: Path,
) -> dict[str, Any]:
    steps = trajectory["steps"]
    reference_literals = None  # kept explicit: this audit reads no answer text beyond file paths
    del reference_literals
    first_refusal = None
    first_unparseable = None
    applied: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        observation = step.get("observation") or ""
        if observation.startswith("Tool error") and first_refusal is None:
            previous = steps[index - 1] if index else None
            first_refusal = {
                "step": index + 1,
                "action": step["action"]["kind"],
                "arguments": step["action"]["arguments"],
                "answered": None
                if previous is None
                else {
                    "kind": previous["action"]["kind"],
                    "arguments": previous["action"]["arguments"],
                    "observation_head": (previous.get("observation") or "").splitlines()[:1],
                },
                "refusal": observation,
            }
        if (
            first_unparseable is None
            and "unparseable" in observation
            and step["action"]["kind"] in {"replace_text", "replace_lines"}
        ):
            first_unparseable = {
                "step": index + 1,
                "action": step["action"]["kind"],
                "arguments": step["action"]["arguments"],
                "classification": classify_unparseable(
                    trajectory, step, source_root, gold_paths
                ),
                "refusal": observation,
            }
        if step["action"]["kind"] in {"replace_text", "replace_lines"} and observation.startswith(
            "Updated "
        ):
            applied.append(
                {
                    "step": index + 1,
                    "action": step["action"]["kind"],
                    "path": step["action"]["arguments"].get("path"),
                    "arguments": step["action"]["arguments"],
                }
            )
    verifier = trajectory.get("verifier") or {}
    return {
        "task_id": trajectory["task_id"],
        "condition": trajectory["policy"]["metadata"].get("diagnostic_condition"),
        "repetition": trajectory["repetition"],
        "seed": trajectory["seed"],
        "steps": trajectory["reward"]["steps"],
        "changed_files": list(trajectory["changed_files"]),
        "failed_nodes": [node.split("::")[-1] for node in verifier.get("failed_nodes") or []],
        "pass_to_pass_regressed": verifier.get("pass_to_pass_regressed"),
        "edits_in_a_repair_file": sorted(
            {entry["path"] for entry in applied if entry["path"] in gold_paths}
        ),
        "edits_outside_the_repair": sorted(
            {entry["path"] for entry in applied if entry["path"] not in gold_paths}
        ),
        "first_refusal": first_refusal,
        "first_unparseable": first_unparseable,
        "applied_edits": applied,
    }


def gold_paths_by_task(path: Path) -> dict[str, set[str]]:
    patches = json.loads(path.read_text(encoding="utf-8"))
    return {
        task_id: set(re.findall(r"^\+\+\+ b/(.+)$", patch, re.M))
        for task_id, patch in patches.items()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--gold-patches", required=True)
    parser.add_argument("--source-root", default="work/private/swe-gym-source-cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--text", help="also write the human-readable dump here")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gold = gold_paths_by_task(Path(args.gold_patches))
    source_root = Path(args.source_root)
    records = [
        audit(
            trajectory,
            gold_paths=gold.get(trajectory["task_id"], set()),
            source_root=source_root,
        )
        for trajectory in jsonl(Path(args.trajectories))
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "trials": len(records),
        "first_refusal_kinds": {},
        "unparseable_classifications": {},
        "applied_edit_trials": sum(bool(row["applied_edits"]) for row in records),
        "trials_editing_outside_the_repair": sum(
            bool(row["edits_outside_the_repair"]) for row in records
        ),
    }
    for row in records:
        refusal = row["first_refusal"]
        if refusal is None:
            key = "no refusal"
        elif REREAD in refusal["refusal"]:
            key = "repeated an already-read file"
        elif "read_file failed" in refusal["refusal"]:
            key = "read a path that does not exist"
        elif "exactly one match" in refusal["refusal"]:
            key = "replace_text found no match"
        elif "do not repeat" in refusal["refusal"]:
            key = "repeated a refused search"
        else:
            key = "other"
        summary["first_refusal_kinds"][key] = summary["first_refusal_kinds"].get(key, 0) + 1
        if row["first_unparseable"] is not None:
            entry = row["first_unparseable"]["classification"]
            if entry["first_line_unindented"]:
                summary["unparseable_classifications"]["first_line_unindented"] = (
                    summary["unparseable_classifications"].get("first_line_unindented", 0) + 1
                )
            key = f"target_range_{entry['target_range']}"
            summary["unparseable_classifications"][key] = (
                summary["unparseable_classifications"].get(key, 0) + 1
            )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.text:
        lines: list[str] = []
        for row in records:
            lines.append("=" * 90)
            lines.append(
                f"{row['task_id']} {row['condition']} r{row['repetition']} steps={row['steps']} "
                f"changed={row['changed_files']} still_failing={row['failed_nodes']} "
                f"p2p_regressed={row['pass_to_pass_regressed']}"
            )
            if row["first_refusal"]:
                refusal = row["first_refusal"]
                lines.append(f"  first refusal at step {refusal['step']}: {refusal['action']}")
                lines.append(f"    answered: {json.dumps(refusal['answered'], ensure_ascii=False)[:400]}")
                lines.append("    refusal:")
                lines.extend(f"      > {line}" for line in refusal["refusal"][:600].splitlines())
            if row["first_unparseable"]:
                entry = row["first_unparseable"]
                lines.append(
                    f"  first unparseable at step {entry['step']}: {entry['action']} "
                    f"-> {json.dumps(entry['classification'], ensure_ascii=False)}"
                )
                lines.append(
                    f"    arguments: {json.dumps(entry['arguments'], ensure_ascii=False)[:600]}"
                )
                lines.extend(
                    f"      > {line}" for line in entry["refusal"][:400].splitlines()
                )
            for entry in row["applied_edits"]:
                lines.append(
                    f"  applied at step {entry['step']}: {entry['path']} "
                    f"{json.dumps({k: v for k, v in entry['arguments'].items() if k != 'new'}, ensure_ascii=False)}"
                )
        Path(args.text).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
