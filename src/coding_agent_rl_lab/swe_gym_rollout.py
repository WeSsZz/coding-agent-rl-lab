from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from .contracts import CodingTask, DatasetSplit, Trajectory
from .docker_environment import DockerSandboxConfig
from .model_policy import OpenAICompatiblePolicy, OpenAICompatiblePolicyConfig
from .policies import Policy
from .providers import DockerSandboxProvider
from .rollout import (
    RolloutCollector,
    build_report,
    read_trajectories,
    write_report,
    write_trajectories,
)
from .swe_gym import SWEGymAdapterConfig, SWEGymTaskAdapter, audited_swe_gym_test_command
from .swe_gym_smoke import (
    PINNED_DEVELOPMENT_ROWS,
    PINNED_INSTANCE_IDS,
    SWE_GYM_TASK_SET_CHOICES,
    dataset_split_for_task_set,
    load_or_download_pinned_rows,
    pinned_rows_for_task_set,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect answer-free model rollouts on SWE-Gym")
    parser.add_argument("--model", required=True, help="Model id exposed by the vLLM server")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--repetitions", type=_positive_int, default=1)
    parser.add_argument("--task-set", choices=SWE_GYM_TASK_SET_CHOICES, default="all")
    parser.add_argument(
        "--task-id",
        action="append",
        choices=PINNED_INSTANCE_IDS,
        default=[],
        help="Exact pinned instance id to run; repeat for multiple tasks.",
    )
    parser.add_argument(
        "--task-count",
        type=_positive_int,
        default=None,
        help=f"Number of pinned development tasks to run (max {len(PINNED_DEVELOPMENT_ROWS)})",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        default=24,
        help="Tool-step budget per trial; a patch needs reading, editing, and repair time",
    )
    parser.add_argument(
        "--context-window-tokens",
        type=_positive_int,
        default=None,
        help=(
            "Served --max-model-len; refuse an oversized prompt before sending it "
            "instead of recording an HTTP 400 as a policy failure"
        ),
    )
    parser.add_argument("--test-timeout-seconds", type=_positive_float, default=900.0)
    parser.add_argument("--rows-cache", default="work/swe-gym-development-rows.jsonl")
    parser.add_argument("--output", default="work/swe-gym-model-report.json")
    parser.add_argument("--trajectories", default="work/swe-gym-model-trajectories.jsonl")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse compatible completed trajectories from the trajectory checkpoint.",
    )
    return parser


def collect_trajectories_incrementally(
    tasks: Sequence[CodingTask],
    policy: Policy,
    collector: RolloutCollector,
    *,
    repetitions: int,
    base_seed: int,
    existing_trajectories: Sequence[Trajectory] = (),
    on_progress: Callable[[tuple[Trajectory, ...], int], None] | None = None,
) -> tuple[Trajectory, ...]:
    planned = len(tasks) * repetitions
    expected: dict[tuple[str, int], int] = {}
    expected_order: list[tuple[str, int]] = []
    for task_index, task in enumerate(tasks):
        task_base_seed = base_seed + task_index * repetitions * 10_000
        for repetition in range(1, repetitions + 1):
            key = (task.task_id, repetition)
            expected[key] = task_base_seed + (repetition - 1) * 10_000
            expected_order.append(key)

    trajectories_by_key: dict[tuple[str, int], Trajectory] = {}
    for trajectory in existing_trajectories:
        key = (trajectory.task_id, trajectory.repetition)
        if key in trajectories_by_key:
            raise ValueError(f"duplicate trajectory checkpoint entry: {key[0]} repetition {key[1]}")
        if key not in expected:
            raise ValueError(f"trajectory checkpoint entry is outside this run: {key[0]} repetition {key[1]}")
        if trajectory.seed != expected[key]:
            raise ValueError(f"trajectory checkpoint seed mismatch for {key[0]} repetition {key[1]}")
        if trajectory.policy != policy.manifest:
            raise ValueError(f"trajectory checkpoint policy mismatch for {key[0]} repetition {key[1]}")
        trajectories_by_key[key] = trajectory

    def ordered_trajectories() -> tuple[Trajectory, ...]:
        return tuple(
            trajectories_by_key[key]
            for key in expected_order
            if key in trajectories_by_key
        )

    if on_progress is not None:
        on_progress(ordered_trajectories(), planned)
    for task_index, task in enumerate(tasks):
        task_base_seed = base_seed + task_index * repetitions * 10_000
        for repetition in range(1, repetitions + 1):
            key = (task.task_id, repetition)
            if key in trajectories_by_key:
                continue
            trajectory = collector.collect(
                task,
                policy,
                repetition=repetition,
                seed=task_base_seed + (repetition - 1) * 10_000,
            )
            trajectories_by_key[key] = trajectory
            if on_progress is not None:
                on_progress(ordered_trajectories(), planned)
    return ordered_trajectories()


def write_rollout_checkpoint(
    tasks: tuple[CodingTask, ...],
    trajectories: tuple[Trajectory, ...],
    *,
    repetitions: int,
    task_set: str,
    split: DatasetSplit,
    planned_trial_count: int,
    report_path: Path,
    trajectory_path: Path,
    run_complete: bool,
) -> dict[str, object]:
    report = build_report(tasks, trajectories, repetitions=repetitions)
    report["task_set"] = task_set
    report["dataset_split"] = split.value
    report["selected_task_ids"] = [task.task_id for task in tasks]
    report["planned_trial_count"] = planned_trial_count
    report["completed_trial_count"] = len(trajectories)
    report["run_complete"] = run_complete
    write_trajectories(trajectories, trajectory_path)
    write_report(report, report_path)
    return report


def main() -> None:
    args = build_parser().parse_args()
    available = pinned_rows_for_task_set(args.task_set)
    if args.task_id and args.task_count is not None:
        raise SystemExit("--task-id and --task-count cannot be used together")
    task_count = args.task_count if args.task_count is not None else 1
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
    adapter = SWEGymTaskAdapter(SWEGymAdapterConfig(max_steps=args.max_steps))
    bundles = tuple(
        adapter.adapt(
            row,
            split=split,
            test_command=audited_swe_gym_test_command(row),
        )
        for row in rows
    )
    provider = DockerSandboxProvider(
        {bundle.task.task_id: bundle.environment for bundle in bundles},
        DockerSandboxConfig(
            memory_limit="4g",
            cpu_limit=2.0,
            pids_limit=512,
            startup_timeout_seconds=180.0,
            command_timeout_seconds=120.0,
            test_timeout_seconds=args.test_timeout_seconds,
        ),
    )
    policy = OpenAICompatiblePolicy(
        OpenAICompatiblePolicyConfig(
            model=args.model,
            api_base=args.api_base,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            context_window_tokens=args.context_window_tokens,
        )
    )
    collector = RolloutCollector(provider)
    tasks = tuple(bundle.task for bundle in bundles)
    report_path = project_root / args.output
    trajectory_path = project_root / args.trajectories
    try:
        existing_trajectories = read_trajectories(trajectory_path) if args.resume else ()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    def checkpoint(trajectories: tuple[Trajectory, ...], planned: int) -> None:
        write_rollout_checkpoint(
            tasks,
            trajectories,
            repetitions=args.repetitions,
            task_set=args.task_set,
            split=split,
            planned_trial_count=planned,
            report_path=report_path,
            trajectory_path=trajectory_path,
            run_complete=False,
        )

    trajectories = collect_trajectories_incrementally(
        tasks,
        policy,
        collector,
        repetitions=args.repetitions,
        base_seed=args.seed,
        existing_trajectories=existing_trajectories,
        on_progress=checkpoint,
    )
    report = write_rollout_checkpoint(
        tasks,
        trajectories,
        repetitions=args.repetitions,
        task_set=args.task_set,
        split=split,
        planned_trial_count=len(tasks) * args.repetitions,
        report_path=report_path,
        trajectory_path=trajectory_path,
        run_complete=True,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"trajectory: {trajectory_path}")


if __name__ == "__main__":
    main()
