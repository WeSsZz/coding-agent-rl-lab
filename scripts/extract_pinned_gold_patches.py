"""Extract the gold patches of pinned SWE-Gym rows into a small, answer-bearing JSON file.

Two things keep the answers out of reach of a rollout, on purpose:

* the pinned row cache under `work/` has `patch` and `hints_text` removed, so a task built from it
  can never carry the fix; and
* the rollout VM has no `pyarrow`, so it cannot read the published train shard either.

That leaves the shard readable only where pyarrow is installed, which is where this script runs. It
writes the JSON that `build_diagnostic_oracle.py` (window positions) and `check_pinned_gold.py`
(environment self-check) consume. Both treat it as an oracle input: it may position an assist and
prove the environment can tell a broken repository from a repaired one, and it must never reach a
prompt, a task, or a trajectory.

The shard is the same file the rows API reads, so it is fetched once by
`python -m coding_agent_rl_lab.swe_gym_sft --download-pinned-train` into
`work/private/swe-gym-train-rows.parquet` and reused here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coding_agent_rl_lab.swe_gym_smoke import PINNED_INSTANCE_IDS  # noqa: E402


def extract(parquet: Path, task_ids: Sequence[str]) -> dict[str, str]:
    try:
        import pyarrow.parquet as parquet_module
    except ModuleNotFoundError as exc:  # the rollout host has no pyarrow, by design
        raise SystemExit(
            "pyarrow is required to read the published shard: python -m pip install pyarrow"
        ) from exc

    table = parquet_module.read_table(parquet)
    wanted = set(task_ids)
    patches: dict[str, str] = {}
    for row in table.to_pylist():
        instance_id = row.get("instance_id")
        if instance_id not in wanted:
            continue
        patch = row.get("patch")
        if isinstance(patch, str) and patch.strip():
            patches[instance_id] = patch
    missing = sorted(wanted - set(patches))
    if missing:
        raise SystemExit(f"no gold patch in {parquet} for: {', '.join(missing)}")
    return {task_id: patches[task_id] for task_id in task_ids}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default="work/private/swe-gym-train-rows.parquet")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task_ids = tuple(args.task_id) if args.task_id else PINNED_INSTANCE_IDS
    unknown = [task_id for task_id in task_ids if task_id not in PINNED_INSTANCE_IDS]
    if unknown:
        raise SystemExit(f"not a pinned instance id: {', '.join(unknown)}")
    patches = extract(Path(args.parquet), task_ids)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(patches, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for task_id, patch in patches.items():
        files = [line[6:] for line in patch.splitlines() if line.startswith("+++ b/")]
        print(f"{task_id}: {len(patch.splitlines())} patch lines, {len(files)} files")
    print(f"wrote {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
