"""Build the frozen auxiliary inputs for the A/B/C model-bottleneck diagnostic.

The diagnostic gives the policy the same task under three conditions:

* ``A`` - the stock prompt and tools only.
* ``B`` - A plus the *implementation file paths* the reviewed repair touches.
* ``C`` - B plus a real, numbered source window of those files *at the task's base commit*.

This script produces the B and C text once, mechanically, and records every hash so the run can
be reproduced and audited. Three rules make the auxiliary text safe and honest:

1. It is derived only from the reviewed repair's *positions* and from the base-commit source.
   No added line of the repair is ever rendered, and no explanation of the change is written.
2. The window rule is fixed here, not chosen per task: pad 20 lines around the union of the
   repair's hunks in each file, cap the window at ``--max-lines-per-file`` by shrinking the pad
   first and the tail last, and cap the whole block at ``--max-total-lines``.
3. The text says out loud that no tool produced it, so a reader of the trajectory cannot mistake
   it for a ``read_file`` result the environment never ran.

The output is a JSONL row per (task, condition) carrying the exact message to inject and its
sha256. It is an *oracle* artifact: it must stay out of any model-visible prompt except through the
diagnostic run that intentionally injects it, and it belongs next to the other answer-bearing
files under ``work/private/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

RAW_SOURCE_URL = "https://raw.githubusercontent.com/getmoto/moto/{commit}/{path}"

HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")

FILE_LIST_HEADER = (
    "[diagnostic auxiliary input - oracle-file condition]\n"
    "This block was not produced by a tool call: the environment did not read or search "
    "anything for you. It is background information supplied by the diagnostic harness.\n"
    "The reviewed repair for this issue touches these implementation files:\n"
)

WINDOW_HEADER = (
    "[diagnostic auxiliary input - oracle-context condition]\n"
    "This block was not produced by a tool call: the environment did not read or search "
    "anything for you. It is background information supplied by the diagnostic harness.\n"
    "The reviewed repair for this issue touches these implementation files:\n"
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def gold_paths(patch: str) -> list[str]:
    """Implementation files the reviewed repair touches, in patch order."""

    paths: list[str] = []
    for line in patch.splitlines():
        if not line.startswith("+++ "):
            continue
        value = line[4:].strip().split("\t", 1)[0]
        if value in {"/dev/null", "dev/null"}:
            continue
        if value.startswith("b/"):
            value = value[2:]
        if value not in paths:
            paths.append(value)
    return paths


def patch_hunks(patch: str) -> dict[str, list[tuple[int, int]]]:
    """``(start, count)`` of every hunk on the **old** side, grouped by implementation file.

    The window renders the file as it exists *before* the repair, so it must be anchored on the old
    side (`-a,b`). The new side (`+c,d`) belongs to a file that does not exist yet in the container:
    where a repair inserts lines earlier in the same file the two sides disagree, and
    `getmoto__moto-7365`'s third hunk is old 385-396 against new 390-396 - a window built from the
    new side with no padding would miss five lines of the region it is meant to show.
    """

    hunks: dict[str, list[tuple[int, int]]] = {}
    current: str | None = None
    for line in patch.splitlines():
        if line.startswith("+++ "):
            value = line[4:].strip().split("\t", 1)[0]
            if value.startswith("b/"):
                value = value[2:]
            current = None if value in {"/dev/null", "dev/null"} else value
            if current is not None:
                hunks.setdefault(current, [])
            continue
        match = HUNK_HEADER.match(line)
        if match and current is not None:
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) is not None else 1
            hunks[current].append((start, max(count, 1)))
    return hunks


def render_window(
    lines: list[str],
    spans: list[tuple[int, int]],
    *,
    pad: int,
    max_lines: int,
) -> tuple[str, dict[str, Any]]:
    """Numbered view of every region the repair touches, plus as much padding as fits.

    The first version of this took the *union* from the first hunk to the last and truncated the
    tail when the result exceeded ``max_lines``. On a file whose hunks sit 2800 lines apart that
    anchored the window on the first hunk and silently dropped the region the failure was actually
    about - `getmoto__moto-7514`'s `models.py` showed lines 1-80 of 2947 while the repair changes
    lines 2861-2926, and `getmoto__moto-7365` showed one of its three hunks.

    So the rule is now: one interval per hunk, merged when they overlap, padding surrendered first
    and **never** a hunk. A file whose hunks alone exceed the cap keeps them anyway and says so.
    """

    total = len(lines)
    if total == 0:
        return "", {
            "intervals": [],
            "first_line": 0,
            "last_line": 0,
            "truncated": False,
            "padding": 0,
        }

    def intervals_for(padding: int) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, count in sorted(spans):
            first = max(1, start - padding)
            last = min(total, start + count - 1 + padding)
            if merged and first <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], last))
            else:
                merged.append((first, last))
        return merged

    def total_lines(intervals: list[tuple[int, int]]) -> int:
        return sum(last - first + 1 for first, last in intervals)

    padding = pad
    while padding > 0 and total_lines(intervals_for(padding)) > max_lines:
        padding -= 1
    intervals = intervals_for(padding)
    over_cap = total_lines(intervals) > max_lines
    rendered: list[str] = []
    for index, (first, last) in enumerate(intervals):
        if index:
            previous = intervals[index - 1][1]
            skipped = first - previous - 1
            rendered.append(
                f"[lines {previous + 1}-{first - 1} not shown: the repair does not touch them "
                f"({skipped} lines)]"
            )
        rendered.extend(f"{number}: {lines[number - 1]}" for number in range(first, last + 1))
    return "\n".join(rendered), {
        "intervals": [{"first_line": first, "last_line": last} for first, last in intervals],
        "first_line": intervals[0][0],
        "last_line": intervals[-1][1],
        "shown_lines": total_lines(intervals),
        "truncated": over_cap,
        "padding": padding,
    }


def added_lines(patch: str, *, minimum_length: int = 12) -> list[str]:
    """The repair's own added lines, stripped, long enough to be a real disclosure."""

    seen: list[str] = []
    for line in patch.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        candidate = line[1:].strip()
        if len(candidate) >= minimum_length and candidate not in seen:
            seen.append(candidate)
    return seen


def added_line_overlap(
    block: str,
    patch: str,
    windows_by_path: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Added-line text that the block also shows, with where the base file already had it.

    A window is the file as it exists at the base commit, so it can legitimately contain a string
    the repair adds *elsewhere* (the 7514 repair duplicates a helper's body into its caller). That
    is a disclosure worth measuring rather than hiding, and it is recorded instead of trimmed:
    trimming would make the window stop being real base-commit source, which is the stronger
    guarantee. Every hit is listed with its base-file line so a reader can check it.
    """

    overlaps: list[dict[str, Any]] = []
    for candidate in added_lines(patch):
        if candidate not in block:
            continue
        locations = []
        for path, window in windows_by_path.items():
            for index, line in enumerate(window["lines"], start=1):
                if any(
                    first <= index <= last for first, last in window["intervals"]
                ) and candidate in line:
                    locations.append({"path": path, "base_line": index})
        overlaps.append({"text": candidate, "in_window_at": locations})
    return overlaps


def file_list_block(paths: list[str]) -> str:
    return FILE_LIST_HEADER + "".join(f"  {path}\n" for path in paths)


def window_block(paths: list[str], entries: list[dict[str, Any]]) -> str:
    """Condition C: the same file list as B, plus a numbered base-commit window where one exists.

    The path list is unconditional, so C is always a superset of B. A file whose source could not be
    read at the base commit therefore still gets named - dropping it would make C quietly weaker
    than B instead of stronger.
    """

    parts = [WINDOW_HEADER]
    parts.append("".join(f"  {path}\n" for path in paths))
    parts.append(
        "Below is the real base-commit source of those files at the lines the failure is about, "
        "with the file's own absolute line numbers. It is the pre-fix code as it exists in the "
        "task snapshot; it is not a diff, it does not contain the fix, and the fix is not "
        "described.\n"
    )
    for entry in entries:
        spans = ", ".join(
            f"{interval['first_line']}-{interval['last_line']}"
            for interval in entry["intervals"]
        )
        parts.append(
            f"\n----- {entry['path']} (base commit {entry['base_commit'][:12]}, "
            f"lines {spans} of {entry['total_lines']})\n"
        )
        parts.append(entry["rendered"] + "\n")
    return "".join(parts)


def fetch(url: str, *, attempts: int = 4) -> str | None:
    for attempt in range(1, attempts + 1):
        try:
            return urllib.request.urlopen(url, timeout=30).read().decode("utf-8")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == attempts:
                print(f"    gave up after {attempts}: {type(exc).__name__}: {exc}", flush=True)
                return None
            time.sleep(2 * attempt)
    return None


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["instance_id"]: row for row in rows}


def load_gold(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in payload.items()
    ):
        raise ValueError("gold patch file must map instance_id to patch text")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-cache", default="work/swe-gym-development-rows.jsonl")
    parser.add_argument("--gold-patches", required=True)
    parser.add_argument("--source-root", default="work/private/swe-gym-source-cache")
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--pad", type=int, default=20)
    parser.add_argument("--max-lines-per-file", type=int, default=80)
    parser.add_argument("--max-total-lines", type=int, default=200)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--no-fetch", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_rows(Path(args.rows_cache))
    gold = load_gold(Path(args.gold_patches))
    source_root = Path(args.source_root)
    records: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "abc-diagnostic-auxiliary-inputs",
        "generated_from": {
            "rows_cache": str(Path(args.rows_cache).resolve()),
            "rows_cache_sha256": hashlib.sha256(Path(args.rows_cache).read_bytes()).hexdigest(),
            "gold_patches": str(Path(args.gold_patches).resolve()),
            "gold_patches_sha256": hashlib.sha256(Path(args.gold_patches).read_bytes()).hexdigest(),
            "source_root": str(source_root.resolve()),
            "raw_source_url": RAW_SOURCE_URL,
        },
        "window_rule": {
            "hunk_side": (
                "old: the window renders the file as it exists before the repair, so it is anchored "
                "on each hunk's `-a,b` coordinates, never the `+c,d` ones"
            ),
            "anchor": "one interval per repair hunk, merged when they overlap",
            "pad_lines": args.pad,
            "max_lines_per_file": args.max_lines_per_file,
            "max_total_lines": args.max_total_lines,
            "cap_order": "shrink padding first, truncate the tail last",
            "rendering": "numbered `N: text`, the file's own absolute line numbers",
        },
        "tasks": [],
        "run_complete": False,
    }
    for task_id in args.task_id:
        if task_id not in rows:
            raise SystemExit(f"{task_id} is not in {args.rows_cache}")
        if task_id not in gold:
            raise SystemExit(f"{task_id} has no gold patch in {args.gold_patches}")
        row = rows[task_id]
        patch = gold[task_id]
        base_commit = row["base_commit"]
        paths = gold_paths(patch)
        hunks = patch_hunks(patch)
        entries: list[dict[str, Any]] = []
        windows_by_path: dict[str, dict[str, Any]] = {}
        total_lines = 0
        skipped: list[dict[str, str]] = []
        for path in paths:
            spans = hunks.get(path)
            if not spans:
                skipped.append({"path": path, "reason": "no hunk in this file's patch section"})
                continue
            cached = source_root / task_id / path
            if cached.is_file():
                body = cached.read_text(encoding="utf-8")
                origin = "cache"
            elif args.no_fetch:
                origin = "missing"
                body = None
            else:
                body = fetch(RAW_SOURCE_URL.format(commit=base_commit, path=path))
                origin = "fetched"
                if body is not None:
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    cached.write_text(body, encoding="utf-8", newline="")
            if body is None:
                skipped.append({"path": path, "reason": "source unavailable at the base commit"})
                continue
            lines = body.splitlines()
            rendered, geometry = render_window(
                lines,
                spans,
                pad=args.pad,
                max_lines=args.max_lines_per_file,
            )
            if total_lines and total_lines + geometry["shown_lines"] > args.max_total_lines:
                skipped.append({"path": path, "reason": "total line budget reached"})
                continue
            total_lines += geometry["shown_lines"]
            entries.append(
                {
                    "path": path,
                    "base_commit": base_commit,
                    "source_origin": origin,
                    "source_sha256": _sha256_text(body),
                    "total_lines": len(lines),
                    **geometry,
                    "hunks": [{"old_start": start, "old_count": count} for start, count in spans],
                    "covers_every_hunk": True,
                    "rendered": rendered,
                    "window_sha256": _sha256_text(rendered),
                }
            )
            windows_by_path[path] = {
                "intervals": [
                    (interval["first_line"], interval["last_line"])
                    for interval in geometry["intervals"]
                ],
                "lines": lines,
            }
        file_block = file_list_block(paths)
        context_block = window_block(paths, [{**entry, "base_commit": base_commit} for entry in entries])
        window_overlap = added_line_overlap(context_block, patch, windows_by_path)
        path_overlap = added_line_overlap(file_block, patch, windows_by_path)
        if window_overlap or path_overlap:
            print(
                f"  disclosure: {task_id} B shows {len(path_overlap)} and C shows "
                f"{len(window_overlap)} added-line strings that also exist in the base source",
                flush=True,
            )
        records.append(
            {
                "task_id": task_id,
                "base_commit": base_commit,
                "condition": "A",
                "auxiliary_message": "",
                "auxiliary_sha256": _sha256_text(""),
                "auxiliary_chars": 0,
            }
        )
        records.append(
            {
                "task_id": task_id,
                "base_commit": base_commit,
                "condition": "B",
                "implementation_paths": paths,
                "auxiliary_message": file_block,
                "auxiliary_sha256": _sha256_text(file_block),
                "auxiliary_chars": len(file_block),
            }
        )
        records.append(
            {
                "task_id": task_id,
                "base_commit": base_commit,
                "condition": "C",
                "implementation_paths": paths,
                "windows": [
                    {key: value for key, value in entry.items() if key != "rendered"}
                    for entry in entries
                ],
                "auxiliary_message": context_block,
                "auxiliary_sha256": _sha256_text(context_block),
                "auxiliary_chars": len(context_block),
                "added_line_disclosure": window_overlap,
            }
        )
        uncovered = [
            {"path": path, "old_start": start, "old_end": start + count - 1}
            for path, spans in hunks.items()
            for start, count in spans
            if not any(
                first <= start and start + count - 1 <= last
                for first, last in windows_by_path.get(path, {}).get("intervals", [])
            )
        ]
        manifest["tasks"].append(
            {
                "task_id": task_id,
                "base_commit": base_commit,
                "implementation_paths": paths,
                "files_with_windows": [entry["path"] for entry in entries],
                "window_line_total": total_lines,
                "skipped": skipped,
                "hunks_not_covered": uncovered,
                "hunks_not_covered_count": len(uncovered),
                "auxiliary_sha256": {
                    "A": _sha256_text(""),
                    "B": _sha256_text(file_block),
                    "C": _sha256_text(context_block),
                },
                "added_line_disclosure": {
                    "rule": (
                        "stripped added lines of the repair with >=12 characters that the block "
                        "also shows, with the base-file line that already carried them"
                    ),
                    "minimum_length": 12,
                    "B": path_overlap,
                    "C": window_overlap,
                },
            }
        )
        print(
            f"{task_id}: {len(paths)} implementation paths, {len(entries)} windows, "
            f"{total_lines} lines, skipped {len(skipped)}",
            flush=True,
        )
    manifest["run_complete"] = True
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest["output"] = str(output.resolve())
    manifest["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "sha256": manifest["output_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
