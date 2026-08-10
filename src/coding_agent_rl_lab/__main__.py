from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import evaluate
from .rollout import write_report, write_trajectories


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verifier-first Coding Agent RL Lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("smoke", help="Run the answer-bearing reference pipeline check")
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a deterministic M0 policy")
    evaluate_parser.add_argument("--policy", choices=("noop", "reference"), default="noop")
    evaluate_parser.add_argument("--repetitions", type=int, default=1)
    evaluate_parser.add_argument("--output", default="work/evaluation-report.json")
    evaluate_parser.add_argument("--trajectories", default="work/trajectories.jsonl")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.command == "smoke":
        report, trajectories = evaluate(project_root, policy_name="reference", repetitions=3)
        write_report(report, project_root / "work" / "smoke-report.json")
        write_trajectories(trajectories, project_root / "work" / "smoke-trajectories.jsonl")
    else:
        report, trajectories = evaluate(
            project_root,
            policy_name=args.policy,
            repetitions=args.repetitions,
        )
        write_report(report, args.output)
        write_trajectories(trajectories, args.trajectories)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
