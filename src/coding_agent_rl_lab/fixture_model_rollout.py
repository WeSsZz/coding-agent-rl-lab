from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .evaluation import load_builtin_tasks
from .model_policy import OpenAICompatiblePolicy, OpenAICompatiblePolicyConfig
from .providers import LocalFixtureEnvironmentProvider
from .rollout import RolloutCollector, build_report, write_report, write_trajectories


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect answer-free model rollouts on trusted curriculum fixtures"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--repetitions", type=_positive_int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=_positive_int, default=1024)
    parser.add_argument("--max-steps", type=_positive_int, default=8)
    parser.add_argument("--output", default="work/fixture-model-report.json")
    parser.add_argument("--trajectories", default="work/fixture-model-trajectories.jsonl")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    tasks = tuple(replace(task, max_steps=args.max_steps) for task in load_builtin_tasks(project_root))
    policy = OpenAICompatiblePolicy(
        OpenAICompatiblePolicyConfig(
            model=args.model,
            api_base=args.api_base,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )
    collector = RolloutCollector(LocalFixtureEnvironmentProvider(project_root))
    trajectories = tuple(
        trajectory
        for task_index, task in enumerate(tasks)
        for trajectory in collector.collect_repetitions(
            task,
            policy,
            repetitions=args.repetitions,
            base_seed=args.seed + task_index * args.repetitions * 10_000,
        )
    )
    report = build_report(tasks, trajectories, repetitions=args.repetitions)
    write_report(report, project_root / args.output)
    write_trajectories(trajectories, project_root / args.trajectories)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"trajectory: {project_root / args.trajectories}")


if __name__ == "__main__":
    main()
