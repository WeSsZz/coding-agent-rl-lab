from __future__ import annotations

import json
import statistics
import uuid
from pathlib import Path
from typing import Any

from .contracts import (
    ActionKind,
    CodingTask,
    PolicyDecision,
    RewardVector,
    TestResult,
    Trajectory,
    TrajectoryStep,
)
from .policies import Policy
from .providers import EnvironmentProvider
from .reward_shaping import build_training_reward, verifier_breakdown


class RolloutCollector:
    def __init__(self, environment_provider: EnvironmentProvider) -> None:
        self.environment_provider = environment_provider

    def collect(self, task: CodingTask, policy: Policy, *, repetition: int, seed: int) -> Trajectory:
        environment = self.environment_provider.create(task)
        steps: list[TrajectoryStep] = []
        terminal_test_result: TestResult | None = None
        edited = False
        verifier_run_after_patch = False
        try:
            initial_observation = environment.reset(task)
            baseline_passed = bool(environment.baseline_result and environment.baseline_result.passed)
            while len(steps) < task.max_steps:
                raw_decision = policy.next_action(
                    task,
                    steps,
                    seed=seed + len(steps),
                    initial_observation=initial_observation,
                )
                decision = (
                    raw_decision
                    if isinstance(raw_decision, PolicyDecision)
                    else PolicyDecision(action=raw_decision)
                )
                if edited and decision.action.kind in {ActionKind.RUN_TESTS, ActionKind.FINISH}:
                    verifier_run_after_patch = True
                result = environment.step(decision.action)
                if (
                    decision.action.kind in {ActionKind.REPLACE_TEXT, ActionKind.REPLACE_LINES}
                    and result.observation.startswith("Updated ")
                ):
                    edited = True
                steps.append(
                    TrajectoryStep(
                        sequence=len(steps) + 1,
                        action=decision.action,
                        observation=result.observation,
                        terminated=result.terminated,
                        test_result=result.test_result,
                        violation=decision.violation or result.violation,
                        policy_input=decision.input_messages,
                        policy_output=decision.output_text,
                        policy_metadata=decision.metadata,
                    )
                )
                if result.terminated:
                    if result.violation is None:
                        terminal_test_result = result.test_result
                    break
            final = terminal_test_result or environment.finalize()
            changed_files = environment.changed_files()
            policy_violations = tuple(step.violation for step in steps if step.violation)
            violations = tuple(dict.fromkeys((*environment.violations, *policy_violations)))
            fail_to_pass, pass_to_pass = environment.graded_targets()
            breakdown = verifier_breakdown(
                final,
                fail_to_pass=fail_to_pass,
                pass_to_pass=pass_to_pass,
            )
            training_reward = build_training_reward(
                baseline=environment.baseline_result,
                final=final,
                patch_created=bool(changed_files),
                patch_valid=environment.patch_is_valid(),
                verifier_run_after_patch=verifier_run_after_patch,
                violations=violations,
                reward_version="conservative-v2",
                fail_to_pass=fail_to_pass,
                pass_to_pass=pass_to_pass,
            )
            reward = RewardVector(
                task_success=final.passed and bool(changed_files) and not violations,
                tests_passed=final.passed,
                regression_free=(
                    final.passed
                    if breakdown is None or not breakdown.comparable
                    else breakdown.regression_free
                ),
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
                initial_observation=initial_observation,
                verifier=breakdown,
                training_reward=training_reward.training_reward,
            )
        finally:
            environment.close()

    def collect_repetitions(
        self,
        task: CodingTask,
        policy: Policy,
        *,
        repetitions: int,
        base_seed: int,
        seed_stride: int = 10_000,
    ) -> tuple[Trajectory, ...]:
        """Collect independent trials, each with a fresh environment and seed range."""

        if repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if seed_stride < task.max_steps:
            raise ValueError("seed_stride must be at least task.max_steps")
        return tuple(
            self.collect(
                task,
                policy,
                repetition=repetition,
                seed=base_seed + (repetition - 1) * seed_stride,
            )
            for repetition in range(1, repetitions + 1)
        )


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
            "success_sample_variance": _sample_variance(outcomes),
            "all_trials_passed": bool(outcomes) and all(outcomes),
            "longest_success_streak": _longest_streak(outcomes),
        }
    successes = [trajectory.reward.task_success for trajectory in trajectories]
    scalar_rewards = [trajectory.reward.scalar for trajectory in trajectories]
    training_rewards = [
        trajectory.training_reward
        for trajectory in trajectories
        if trajectory.training_reward is not None
    ]
    return {
        "schema_version": 1,
        "project_stage": (
            "m1_model_rollout_no_training"
            if trajectories and trajectories[0].policy.model
            else "m0_environment_no_training"
        ),
        "policy": trajectories[0].policy.__dict__ if trajectories else None,
        "task_count": len(tasks),
        "repetitions": repetitions,
        "trial_count": len(trajectories),
        "pass_at_1": round(sum(successes) / len(successes), 4) if successes else 0.0,
        "success_sample_variance": _sample_variance(successes),
        "pass_power_3": round(sum(windows) / len(windows), 4) if windows else None,
        "fully_reliable_task_rate": round(
            sum(bool(item["all_trials_passed"]) for item in case_reliability.values()) / len(case_reliability),
            4,
        ) if case_reliability else 0.0,
        "mean_scalar_reward": round(
            sum(scalar_rewards) / len(scalar_rewards),
            4,
        ) if scalar_rewards else 0.0,
        "mean_training_reward": round(
            sum(training_rewards) / len(training_rewards),
            4,
        ) if training_rewards else 0.0,
        "scalar_reward_sample_variance": _sample_variance(scalar_rewards),
        "violation_count": sum(len(item.reward.violations) for item in trajectories),
        "task_reliability": case_reliability,
        "training_performed": False,
    }


def write_trajectories(trajectories: tuple[Trajectory, ...], path: str | Path) -> None:
    target = Path(path)
    content = "\n".join(json.dumps(item.to_dict(), ensure_ascii=False) for item in trajectories)
    _atomic_write_text(target, content + ("\n" if content else ""))


def read_trajectories(path: str | Path) -> tuple[Trajectory, ...]:
    target = Path(path)
    if not target.exists():
        return ()
    trajectories: list[Trajectory] = []
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("trajectory row must be an object")
            trajectories.append(Trajectory.from_dict(value))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid trajectory checkpoint row {line_number}: {exc}") from exc
    return tuple(trajectories)


def write_report(report: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    _atomic_write_text(target, json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def _atomic_write_text(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _longest_streak(values: list[bool]) -> int:
    best = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _sample_variance(values: list[bool] | list[float]) -> float | None:
    if len(values) < 2:
        return None
    return round(statistics.variance(float(value) for value in values), 4)

