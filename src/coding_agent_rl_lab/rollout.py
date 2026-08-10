from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Callable

from .contracts import ActionKind, AgentAction, CodingTask, RewardVector, Trajectory, TrajectoryStep
from .environment import LocalFixtureEnvironment
from .policies import Policy


EnvironmentFactory = Callable[[], LocalFixtureEnvironment]


class RolloutCollector:
    def __init__(self, environment_factory: EnvironmentFactory) -> None:
        self.environment_factory = environment_factory

    def collect(self, task: CodingTask, policy: Policy, *, repetition: int, seed: int) -> Trajectory:
        environment = self.environment_factory()
        steps: list[TrajectoryStep] = []
        try:
            environment.reset(task)
            baseline_passed = bool(environment.baseline_result and environment.baseline_result.passed)
            while len(steps) < task.max_steps:
                try:
                    action = policy.next_action(task, steps)
                except Exception as exc:
                    violation = f"policy_error:{type(exc).__name__}"
                    environment.violations.append(violation)
                    steps.append(
                        TrajectoryStep(
                            sequence=len(steps) + 1,
                            action=AgentAction(ActionKind.FINISH),
                            observation=f"Policy failed closed: {type(exc).__name__}: {str(exc)[:1000]}",
                            terminated=True,
                            violation=violation,
                        )
                    )
                    break
                result = environment.step(action)
                steps.append(
                    TrajectoryStep(
                        sequence=len(steps) + 1,
                        action=action,
                        observation=result.observation,
                        terminated=result.terminated,
                        test_result=result.test_result,
                        violation=result.violation,
                    )
                )
                if result.terminated:
                    break
            final = environment.finalize()
            changed_files = environment.changed_files()
            violations = tuple(dict.fromkeys(environment.violations))
            reward = RewardVector(
                task_success=final.passed and bool(changed_files) and not violations,
                tests_passed=final.passed,
                regression_free=final.passed,
                patch_created=bool(changed_files),
                tool_calls=environment.tool_calls,
                steps=len(steps),
                violations=violations,
            )
            return Trajectory(
                trajectory_id=f"traj-{uuid.uuid4().hex}",
                task_id=task.task_id,
                repetition=repetition,
                seed=seed,
                policy=policy.manifest,
                steps=tuple(steps),
                reward=reward,
                changed_files=changed_files,
                baseline_tests_passed=baseline_passed,
                final_tests_passed=final.passed,
            )
        finally:
            environment.close()


def build_report(
    tasks: tuple[CodingTask, ...],
    trajectories: tuple[Trajectory, ...],
    *,
    repetitions: int,
) -> dict[str, Any]:
    by_task: dict[str, list[Trajectory]] = {task.task_id: [] for task in tasks}
    for trajectory in trajectories:
        by_task.setdefault(trajectory.task_id, []).append(trajectory)
    case_reliability: dict[str, Any] = {}
    windows: list[bool] = []
    for task_id, task_trajectories in sorted(by_task.items()):
        ordered = sorted(task_trajectories, key=lambda item: item.repetition)
        outcomes = [item.reward.task_success for item in ordered]
        windows.extend(all(outcomes[index : index + 3]) for index in range(len(outcomes) - 2))
        case_reliability[task_id] = {
            "trial_count": len(outcomes),
            "success_rate": round(sum(outcomes) / len(outcomes), 4) if outcomes else 0.0,
            "all_trials_passed": bool(outcomes) and all(outcomes),
            "longest_success_streak": _longest_streak(outcomes),
        }
    successes = [trajectory.reward.task_success for trajectory in trajectories]
    return {
        "schema_version": 1,
        "project_stage": "m1_adapter_no_training",
        "policy": trajectories[0].policy.__dict__ if trajectories else None,
        "task_count": len(tasks),
        "repetitions": repetitions,
        "trial_count": len(trajectories),
        "pass_at_1": round(sum(successes) / len(successes), 4) if successes else 0.0,
        "pass_power_3": round(sum(windows) / len(windows), 4) if windows else None,
        "fully_reliable_task_rate": round(
            sum(bool(item["all_trials_passed"]) for item in case_reliability.values()) / len(case_reliability),
            4,
        ) if case_reliability else 0.0,
        "mean_scalar_reward": round(
            sum(item.reward.scalar for item in trajectories) / len(trajectories),
            4,
        ) if trajectories else 0.0,
        "violation_count": sum(len(item.reward.violations) for item in trajectories),
        "task_reliability": case_reliability,
        "training_performed": False,
    }


def write_trajectories(trajectories: tuple[Trajectory, ...], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(item.to_dict(), ensure_ascii=False) for item in trajectories)
    target.write_text(content + ("\n" if content else ""), encoding="utf-8")


def write_report(report: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _longest_streak(values: list[bool]) -> int:
    best = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best
