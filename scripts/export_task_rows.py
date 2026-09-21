"""Export full SWE-Gym rows for named instances as JSONL.

The Ubuntu worker has no `pyarrow`, so it cannot read the official shard itself. This exporter
is therefore how a task reaches the worker: it writes whichever rows were asked for, together
with a machine-checkable summary of what it wrote.

The rows contain the gold `patch`, so the default output path is under `work/private/` and the
script refuses to write an answer-bearing file into a tracked directory unless `--allow-tracked`
is passed explicitly.

Usage:
    python scripts/export_task_rows.py --instance-id conan-io__conan-10975 \
        --instance-id pydantic__pydantic-8947 --output work/private/audit-rows.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHARD = REPOSITORY_ROOT / "work/private/swe-gym-train-rows.parquet"


class ExportError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, default=DEFAULT_SHARD)
    parser.add_argument("--instance-id", action="append", required=True, dest="instance_ids")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-tracked",
        action="store_true",
        help="permit writing an answer-bearing file outside work/private",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("FAILED  pyarrow is required to read the official shard")
        return 1

    wanted = list(dict.fromkeys(arguments.instance_ids))
    rows = [row for row in pq.read_table(arguments.shard).to_pylist() if row["instance_id"] in wanted]
    found = {row["instance_id"] for row in rows}
    missing = [item for item in wanted if item not in found]
    if missing:
        print(f"FAILED  not in the shard: {missing}")
        return 1

    resolved = arguments.output.resolve()
    private_root = (REPOSITORY_ROOT / "work/private").resolve()
    if private_root not in resolved.parents and not arguments.allow_tracked:
        print(f"FAILED  refusing to write an answer-bearing file to {resolved}")
        return 1

    # Keep the requested order so a diff of two exports is stable.
    ordered = sorted(rows, key=lambda row: wanted.index(row["instance_id"]))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as stream:
        for row in ordered:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    print(f"wrote {len(ordered)} row(s) to {resolved}")
    print(f"sha256 {digest}")
    for row in ordered:
        print(
            f"  {row['instance_id']} {row['repo']}@{row['version']} "
            f"f2p={len(row['FAIL_TO_PASS'])} p2p={len(row['PASS_TO_PASS'])}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
