from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import evaluate
from .model import OpenAICompatibleActionModel, OpenAICompatibleConfig
from .rollout import write_report, write_trajectories
from .sandbox import DockerSandboxProvider
from .swe_gym import build_swebench_harness_command, load_swe_gym_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verifier-first Coding Agent RL Lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("smoke", help="Run the answer-bearing reference pipeline check")
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a baseline or model coding policy")
    evaluate_parser.add_argument("--policy", choices=("noop", "reference", "model"), default="noop")
    evaluate_parser.add_argument("--repetitions", type=int, default=1)
    evaluate_parser.add_argument("--output", default="work/evaluation-report.json")
    evaluate_parser.add_argument("--trajectories", default="work/trajectories.jsonl")
    evaluate_parser.add_argument("--base-url", help="OpenAI-compatible API base URL for --policy model")
    evaluate_parser.add_argument("--model", help="Model name for --policy model")
    evaluate_parser.add_argument("--api-key-env", default="CODING_AGENT_API_KEY")
    swe_parser = subparsers.add_parser("swe-plan", help="Validate SWE-Gym JSONL and print an official harness command")
    swe_parser.add_argument("--instances-jsonl", required=True)
    swe_parser.add_argument("--dataset-name", required=True)
    swe_parser.add_argument("--predictions", required=True)
    swe_parser.add_argument("--run-id", required=True)
    swe_parser.add_argument("--max-workers", type=int, default=1)
    swe_parser.add_argument("--cache-level", choices=("none", "base", "env", "instance"), default="env")
    subparsers.add_parser("docker-check", help="Check whether the Docker sandbox runtime is available")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.command == "smoke":
        report, trajectories = evaluate(project_root, policy_name="reference", repetitions=3)
        write_report(report, project_root / "work" / "smoke-report.json")
        write_trajectories(trajectories, project_root / "work" / "smoke-trajectories.jsonl")
    elif args.command == "evaluate":
        action_model = None
        if args.policy == "model":
            if not args.base_url or not args.model:
                raise SystemExit("--policy model requires --base-url and --model")
            config = OpenAICompatibleConfig.from_env(
                base_url=args.base_url,
                model=args.model,
                api_key_env=args.api_key_env,
            )
            action_model = OpenAICompatibleActionModel(config)
        report, trajectories = evaluate(
            project_root,
            policy_name=args.policy,
            repetitions=args.repetitions,
            action_model=action_model,
        )
        write_report(report, args.output)
        write_trajectories(trajectories, args.trajectories)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    elif args.command == "swe-plan":
        instances = load_swe_gym_jsonl(args.instances_jsonl)
        command = build_swebench_harness_command(
            dataset_name=args.dataset_name,
            predictions_path=args.predictions,
            run_id=args.run_id,
            instance_ids=tuple(instance.instance_id for instance in instances),
            max_workers=args.max_workers,
            cache_level=args.cache_level,
        )
        print(
            json.dumps(
                {"validated_instances": len(instances), "command": list(command)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    else:
        available = DockerSandboxProvider().available()
        print(json.dumps({"docker_available": available}, indent=2))
        raise SystemExit(0 if available else 1)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
