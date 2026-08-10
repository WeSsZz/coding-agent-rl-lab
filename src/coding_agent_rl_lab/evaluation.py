from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import CodingTask, Trajectory
from .dataset import load_reference_actions, load_tasks
from .environment import LocalFixtureEnvironment
from .model import StructuredActionModel
from .policies import ModelCodingPolicy, NoOpPolicy, Policy, ReferencePolicy
from .rollout import RolloutCollector, build_report


def load_builtin_tasks(project_root: str | Path) -> tuple[CodingTask, ...]:
    root = Path(project_root)
    return load_tasks(root / "datasets" / "development" / "tasks.jsonl")


def build_policy(
    name: str,
    project_root: str | Path,
    *,
    action_model: StructuredActionModel | None = None,
) -> Policy:
    root = Path(project_root)
    if name == "noop":
        return NoOpPolicy()
    if name == "reference":
        actions = load_reference_actions(root / "datasets" / "development" / "reference_actions.jsonl")
        return ReferencePolicy(actions)
    if name == "model":
        if action_model is None:
            raise ValueError("model policy requires an action model")
        return ModelCodingPolicy(action_model)
    raise ValueError(f"unknown policy: {name}")


def evaluate(
    project_root: str | Path,
    *,
    policy_name: str,
    repetitions: int,
    action_model: StructuredActionModel | None = None,
) -> tuple[dict[str, Any], tuple[Trajectory, ...]]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    root = Path(project_root).resolve()
    tasks = load_builtin_tasks(root)
    policy = build_policy(policy_name, root, action_model=action_model)
    collector = RolloutCollector(lambda: LocalFixtureEnvironment(root))
    trajectories = tuple(
        collector.collect(
            task,
            policy,
            repetition=repetition + 1,
            seed=10_000 * (repetition + 1) + task_index,
        )
        for repetition in range(repetitions)
        for task_index, task in enumerate(tasks)
    )
    return build_report(tasks, trajectories, repetitions=repetitions), trajectories
