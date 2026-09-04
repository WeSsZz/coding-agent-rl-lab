from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from .contracts import CodingTask
from .grpo_environment import build_grpo_prompt_rows
from .swe_gym import SWEGymTaskAdapter, audited_swe_gym_test_command
from .swe_gym_smoke import (
    PINNED_INSTANCE_IDS,
    SWE_GYM_TASK_SET_CHOICES,
    dataset_split_for_task_set,
    load_or_download_pinned_rows,
    pinned_rows_for_task_set,
)


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache pinned SWE-Gym rows and pull their Docker images"
    )
    parser.add_argument(
        "--task-count",
        type=_positive_int,
        default=None,
    )
    parser.add_argument("--task-set", choices=SWE_GYM_TASK_SET_CHOICES, default="all")
    parser.add_argument(
        "--task-id",
        action="append",
        choices=PINNED_INSTANCE_IDS,
        default=[],
        help="Exact pinned instance id to prepare; repeat for multiple tasks.",
    )
    parser.add_argument("--attempts", type=_positive_int, default=3)
    parser.add_argument("--rows-cache", default="work/swe-gym-development-rows.jsonl")
    parser.add_argument("--prompt-output", default="work/swe-gym-grpo-prompts.jsonl")
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Only validate task rows and write answer-free GRPO prompts.",
    )
    return parser


def prepare_images(
    images: Iterable[str],
    *,
    attempts: int,
    runner: CommandRunner = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[int, int]:
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    pulled = 0
    skipped = 0
    for image in images:
        present = runner(
            ("docker", "image", "inspect", image),
            capture_output=True,
            text=True,
            check=False,
        )
        if present.returncode == 0:
            skipped += 1
            print(f"SKIP {image} (already present)", flush=True)
            continue
        last_error = ""
        for attempt in range(1, attempts + 1):
            print(f"PULL {image} (attempt {attempt}/{attempts})", flush=True)
            result = runner(
                ("docker", "pull", "--quiet", image),
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                pulled += 1
                print(f"DONE {image}", flush=True)
                break
            last_error = (result.stderr or result.stdout).strip()
            if attempt < attempts:
                sleeper(float(2 ** (attempt - 1)))
        else:
            raise RuntimeError(f"failed to pull {image}: {last_error}")
    return pulled, skipped


def write_prompt_rows(path: Path, tasks: Iterable[CodingTask]) -> int:
    rows = build_grpo_prompt_rows(tasks)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def main() -> None:
    args = build_parser().parse_args()
    available = pinned_rows_for_task_set(args.task_set)
    if args.task_id and args.task_count is not None:
        raise SystemExit("--task-id and --task-count cannot be used together")
    task_count = args.task_count if args.task_count is not None else len(available)
    if not args.task_id and task_count > len(available):
        raise SystemExit(
            f"--task-count cannot exceed {len(available)} for task set {args.task_set}"
        )
    project_root = Path(__file__).resolve().parents[2]
    try:
        rows = load_or_download_pinned_rows(
            project_root / args.rows_cache,
            limit=None if args.task_id else task_count,
            task_set=args.task_set,
            task_ids=args.task_id,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    split = dataset_split_for_task_set(args.task_set)
    adapter = SWEGymTaskAdapter()
    bundles = tuple(
        adapter.adapt(
            row,
            split=split,
            test_command=audited_swe_gym_test_command(row),
        )
        for row in rows
    )
    prompt_count = write_prompt_rows(
        project_root / args.prompt_output,
        (bundle.task for bundle in bundles),
    )
    if args.skip_images:
        pulled, skipped = 0, 0
    else:
        pulled, skipped = prepare_images(
            (bundle.environment.image for bundle in bundles),
            attempts=args.attempts,
        )
    print(
        f"READY tasks={len(bundles)} prompts={prompt_count} "
        f"pulled={pulled} already_present={skipped}",
        flush=True,
    )


if __name__ == "__main__":
    main()
