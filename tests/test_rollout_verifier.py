from __future__ import annotations

import unittest

from coding_agent_rl_lab.contracts import (
    ActionKind,
    AgentAction,
    CodingTask,
    DatasetSplit,
    PolicyDecision,
    PolicyManifest,
    StepResult,
    TestResult,
)
from coding_agent_rl_lab.rollout import RolloutCollector, build_report


FAIL_TO_PASS = ("tests/test_bug.py::test_decimal",)
PASS_TO_PASS = ("tests/test_existing.py::test_regression",)


def _result(*failed: str, passed: bool = False, stderr: str = "") -> TestResult:
    summary = "\n".join(f"FAILED {item} - assertion" for item in failed)
    return TestResult(
        command=("pytest",),
        passed=passed,
        exit_code=0 if passed else 1,
        stdout=summary,
        stderr=stderr,
        duration_ms=1.0,
    )


def _task() -> CodingTask:
    return CodingTask(
        task_id="task-1",
        issue="decimal arithmetic in update_item",
        fixture_path=None,
        base_commit="0" * 40,
        test_command=("pytest", "--", *FAIL_TO_PASS, *PASS_TO_PASS),
        split=DatasetSplit.DEVELOPMENT,
        provenance="test",
        max_steps=1,
    )


class _ScriptedEnvironment:
    """Minimal CodingEnvironment double that replays one fixed verifier outcome."""

    def __init__(
        self,
        baseline: TestResult,
        final: TestResult,
        changed_files: tuple[str, ...],
    ) -> None:
        self.baseline_result = baseline
        self.last_test_result = baseline
        self.tool_calls = 0
        self.violations: list[str] = []
        self.closed = False
        self.finalize_calls = 0
        self._final = final
        self._changed_files = changed_files

    def reset(self, task: CodingTask) -> str:
        del task
        self.tool_calls += 1
        return f"Baseline verifier result:\n{self.baseline_result.stdout}"

    def step(self, action: AgentAction) -> StepResult:
        del action
        self.tool_calls += 1
        return StepResult("Tool error: unsupported in the double", False, self.last_test_result)

    def finalize(self) -> TestResult:
        self.finalize_calls += 1
        return self._final

    def changed_files(self) -> tuple[str, ...]:
        return self._changed_files

    def patch_is_valid(self) -> bool:
        return bool(self._changed_files)

    def graded_targets(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return FAIL_TO_PASS, PASS_TO_PASS

    @property
    def loop_rejections(self) -> int:
        return 0

    def close(self) -> None:
        self.closed = True


class _FinishPolicy:
    manifest = PolicyManifest(policy_id="finish", version="1", policy_type="test")

    def next_action(self, task, history, *, seed=None, initial_observation=""):
        del task, history, seed, initial_observation
        return AgentAction(ActionKind.FINISH)


class _SingleEnvironmentProvider:
    def __init__(self, environment: _ScriptedEnvironment) -> None:
        self.environment = environment

    def create(self, task: CodingTask) -> _ScriptedEnvironment:
        del task
        return self.environment


class _UnreachableEndpointPolicy:
    """A policy that cannot reach the model and answers with the `finish` fallback."""

    manifest = PolicyManifest(policy_id="unreachable", version="1", policy_type="test")

    def next_action(self, task, history, *, seed=None, initial_observation=""):
        del task, history, seed, initial_observation
        return PolicyDecision(
            action=AgentAction(ActionKind.FINISH),
            metadata={"errors": ["model endpoint request failed: Connection refused"]},
            violation="policy_transport_error",
        )


class RolloutVerifierTests(unittest.TestCase):
    def test_a_policy_that_cannot_reach_the_model_ends_the_episode(self) -> None:
        environment = _ScriptedEnvironment(_result(*FAIL_TO_PASS), _result(*FAIL_TO_PASS), ())
        collector = RolloutCollector(_SingleEnvironmentProvider(environment))

        trajectory = collector.collect(
            _task(),
            _UnreachableEndpointPolicy(),
            repetition=1,
            seed=7,
        )

        self.assertEqual(len(trajectory.steps), 1)
        self.assertTrue(trajectory.steps[0].terminated)
        self.assertEqual(trajectory.steps[0].violation, "policy_transport_error")
        self.assertIn("Connection refused", trajectory.steps[0].observation)
        self.assertEqual(trajectory.reward.violations, ("policy_transport_error",))
        # The fallback `finish` is never executed - `reset` is the only call the double sees -
        # and the verifier still grades whatever the container holds.
        self.assertEqual(environment.tool_calls, 1)
        self.assertEqual(environment.finalize_calls, 1)

    def _collect(
        self,
        final: TestResult,
        *,
        changed_files: tuple[str, ...] = ("src/code.py",),
    ) -> tuple[object, _ScriptedEnvironment]:
        environment = _ScriptedEnvironment(_result(*FAIL_TO_PASS), final, changed_files)
        collector = RolloutCollector(_SingleEnvironmentProvider(environment))
        trajectory = collector.collect(_task(), _FinishPolicy(), repetition=1, seed=7)
        return trajectory, environment

    def test_regression_free_stops_being_a_copy_of_tests_passed(self) -> None:
        trajectory, environment = self._collect(_result(*FAIL_TO_PASS))

        self.assertFalse(trajectory.reward.tests_passed)
        self.assertTrue(trajectory.reward.regression_free)
        self.assertEqual(trajectory.verifier.fail_to_pass_total, 1)
        self.assertEqual(trajectory.verifier.fail_to_pass_resolved, 0)
        self.assertEqual(trajectory.verifier.pass_to_pass_regressed, 0)
        self.assertTrue(environment.closed)

    def test_pass_to_pass_regression_blocks_regression_free(self) -> None:
        trajectory, _ = self._collect(_result("tests/test_existing.py::test_regression"))

        self.assertEqual(trajectory.verifier.pass_to_pass_regressed, 1)
        self.assertFalse(trajectory.reward.regression_free)

    def test_passing_run_resolves_the_declared_target_and_earns_strict_reward(self) -> None:
        trajectory, _ = self._collect(_result(passed=True))

        self.assertTrue(trajectory.reward.task_success)
        self.assertTrue(trajectory.reward.regression_free)
        self.assertEqual(trajectory.verifier.fail_to_pass_resolved, 1)
        self.assertEqual(trajectory.training_reward, 1.0)

    def test_collection_error_is_recorded_and_earns_no_training_reward(self) -> None:
        trajectory, _ = self._collect(_result(stderr="ERROR collecting tests/test_bug.py"))

        self.assertTrue(trajectory.verifier.collection_error)
        self.assertFalse(trajectory.reward.regression_free)
        self.assertEqual(trajectory.training_reward, 0.0)

    def test_report_aggregates_both_reward_scales(self) -> None:
        trajectory, _ = self._collect(_result(passed=True))
        report = build_report((_task(),), (trajectory,), repetitions=1)

        self.assertEqual(report["pass_at_1"], 1.0)
        self.assertEqual(report["mean_training_reward"], 1.0)


if __name__ == "__main__":
    unittest.main()
