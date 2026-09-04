from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .contracts import RewardVector


class RLDatasetError(ValueError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export answer-free model trajectories as agentic RL episodes"
    )
    parser.add_argument("--input", required=True, help="Trajectory v3 JSONL")
    parser.add_argument("--output", default="work/rl-episodes-v1.jsonl")
    parser.add_argument("--report", default="work/rl-episodes-v1-report.json")
    return parser


def load_trajectory_rows(path: str | Path) -> tuple[dict[str, Any], ...]:
    target = Path(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RLDatasetError(f"invalid JSON at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise RLDatasetError(f"trajectory at line {line_number} must be an object")
        trajectory_id = row.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise RLDatasetError(f"trajectory at line {line_number} has no trajectory_id")
        if trajectory_id in seen:
            raise RLDatasetError(f"duplicate trajectory_id at line {line_number}: {trajectory_id}")
        seen.add(trajectory_id)
        rows.append(row)
    if not rows:
        raise RLDatasetError("trajectory dataset must not be empty")
    return tuple(rows)


def export_rl_episodes(
    rows: Iterable[dict[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for row in rows:
        reason = _exclusion_reason(row)
        if reason is not None:
            exclusions[reason] += 1
            continue
        episodes.append(_episode_from_trajectory(row))
    positive_count = sum(bool(item["reward"]["task_success"]) for item in episodes)
    report = {
        "schema_version": 1,
        "episode_schema": "agentic-rl-episode-v1",
        "episode_count": len(episodes),
        "positive_count": positive_count,
        "negative_count": len(episodes) - positive_count,
        "excluded_count": sum(exclusions.values()),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "training_performed": False,
        "intended_use": "online-agentic-rl-input-not-trained-weights",
    }
    return tuple(episodes), report


def write_rl_dataset(
    episodes: tuple[dict[str, Any], ...],
    report: dict[str, Any],
    *,
    output_path: str | Path,
    report_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in episodes)
        + ("\n" if episodes else ""),
        encoding="utf-8",
    )
    report_target = Path(report_path)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _exclusion_reason(row: dict[str, Any]) -> str | None:
    if row.get("schema_version") != 3:
        return "unsupported_trajectory_schema"
    policy = row.get("policy")
    if not isinstance(policy, dict) or not policy.get("model"):
        return "non_model_policy"
    metadata = policy.get("metadata")
    if isinstance(metadata, dict) and metadata.get("contains_answers"):
        return "answer_containing_policy"
    if row.get("baseline_tests_passed") is not False:
        return "invalid_baseline"
    steps = row.get("steps")
    if not isinstance(steps, list) or not steps:
        return "empty_episode"
    reward = row.get("reward")
    violations = reward.get("violations") if isinstance(reward, dict) else None
    if not isinstance(violations, list):
        return "invalid_reward"
    if "policy_transport_error" in violations:
        return "model_transport_failure"
    if _contains_sensitive_key(row):
        return "sensitive_metadata"
    return None


def _episode_from_trajectory(row: dict[str, Any]) -> dict[str, Any]:
    reward_payload = row["reward"]
    reward = RewardVector(
        task_success=bool(reward_payload["task_success"]),
        tests_passed=bool(reward_payload["tests_passed"]),
        regression_free=bool(reward_payload["regression_free"]),
        patch_created=bool(reward_payload["patch_created"]),
        tool_calls=int(reward_payload["tool_calls"]),
        steps=int(reward_payload["steps"]),
        violations=tuple(reward_payload["violations"]),
    )
    turns = tuple(
        {
            "sequence": step["sequence"],
            "messages": step.get("policy_input", []),
            "response": step.get("policy_output"),
            "action": step["action"],
            "observation": step["observation"],
            "violation": step.get("violation"),
            "model_metadata": step.get("policy_metadata", {}),
        }
        for step in row["steps"]
    )
    return {
        "schema_version": 1,
        "episode_id": row["trajectory_id"],
        "task_id": row["task_id"],
        "repetition": row["repetition"],
        "seed": row["seed"],
        "policy": row["policy"],
        "initial_observation": row["initial_observation"],
        "turns": turns,
        "reward": {**reward_payload, "scalar": reward.scalar},
        "changed_files": row.get("changed_files", []),
        "final_tests_passed": bool(row.get("final_tests_passed")),
        "eligible_for_sft": reward.task_success and not reward.violations,
    }


def _contains_sensitive_key(value: Any) -> bool:
    sensitive = {"authorization", "api_key", "password", "secret"}
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in sensitive and child not in (None, "", [], {}):
                return True
            if _contains_sensitive_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_sensitive_key(item) for item in value)
    return False


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    rows = load_trajectory_rows(project_root / args.input)
    episodes, report = export_rl_episodes(rows)
    write_rl_dataset(
        episodes,
        report,
        output_path=project_root / args.output,
        report_path=project_root / args.report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
