from __future__ import annotations

import unittest

from coding_agent_rl_lab.contracts import (
    ActionKind,
    AgentAction,
    PolicyManifest,
    RewardVector,
    TestResult,
    Trajectory,
    TrajectoryStep,
    VerifierBreakdown,
)
from coding_agent_rl_lab.failure_analysis import build_failure_report, classify_trajectory


def _trajectory(
    *,
    violations: tuple[str, ...] = (),
    changed_files: tuple[str, ...] = (),
    errors: tuple[str, ...] = (),
    observation: str = "",
    timed_out: bool = False,
    success: bool = False,
    edit_attempt_failed: bool = False,
    verifier: VerifierBreakdown | None = None,
) -> Trajectory:
    test_result = TestResult(("pytest",), False, None, "", "", 1.0, timed_out)
    step = TrajectoryStep(
        sequence=1,
        action=(
            AgentAction(
                ActionKind.REPLACE_TEXT,
                {"path": "src/code.py", "old": "before", "new": "after"},
            )
            if edit_attempt_failed
            else AgentAction(ActionKind.FINISH)
        ),
        observation=(
            "Tool error: replace_text requires exactly one match, found 0"
            if edit_attempt_failed
            else observation
        ),
        terminated=True,
        test_result=test_result,
        policy_metadata={"errors": list(errors)} if errors else {},
    )
    return Trajectory(
        trajectory_id="trajectory-1",
        task_id="task-1",
        repetition=1,
        seed=123,
        policy=PolicyManifest("policy", "1", "test"),
        steps=(step,),
        reward=RewardVector(
            task_success=success,
            tests_passed=success,
            regression_free=success,
            patch_created=bool(changed_files),
            tool_calls=1,
            steps=1,
            violations=violations,
        ),
        changed_files=changed_files,
        baseline_tests_passed=False,
        final_tests_passed=success,
        initial_observation="baseline failed",
        verifier=verifier,
    )


class FailureAnalysisTests(unittest.TestCase):
    def test_context_overflow_is_separated_from_generic_transport_failure(self) -> None:
        trajectory = _trajectory(
            violations=("policy_transport_error",),
            errors=("maximum context length is 8192 tokens",),
        )

        self.assertEqual(classify_trajectory(trajectory), "context_window_exceeded")

    def test_protected_test_edit_and_timeout_have_actionable_categories(self) -> None:
        protected = _trajectory(
            violations=("invalid_action:EnvironmentError",),
            observation="cannot modify verifier-owned test file: tests/test_bug.py",
        )
        timeout = _trajectory(changed_files=("src/code.py",), timed_out=True)

        self.assertEqual(classify_trajectory(protected), "protected_test_edit")
        self.assertEqual(classify_trajectory(timeout), "verifier_timeout")

    def test_failed_patch_failed_edit_and_no_patch_are_distinct(self) -> None:
        self.assertEqual(
            classify_trajectory(_trajectory(changed_files=("src/code.py",))),
            "patch_failed_verifier",
        )
        self.assertEqual(
            classify_trajectory(_trajectory(edit_attempt_failed=True)),
            "edit_action_failed",
        )
        self.assertEqual(classify_trajectory(_trajectory()), "no_patch")

    def test_report_contains_counts_without_raw_content(self) -> None:
        report = build_failure_report(
            (_trajectory(changed_files=("src/code.py",)), _trajectory())
        )

        self.assertEqual(report["trial_count"], 2)
        self.assertEqual(report["category_counts"], {"patch_failed_verifier": 1, "no_patch": 1})
        self.assertFalse(report["contains_raw_model_or_repository_content"])
        self.assertNotIn("observation", report["trajectories"][0])

    def test_graded_attribution_is_reported_per_trial_and_aggregated(self) -> None:
        report = build_failure_report(
            (
                _trajectory(
                    changed_files=("src/code.py",),
                    verifier=VerifierBreakdown(
                        fail_to_pass_total=2,
                        pass_to_pass_total=3,
                        fail_to_pass_resolved=1,
                        pass_to_pass_regressed=1,
                        failed_nodes=("tests/test_existing.py::test_regression",),
                        node_targets_declared=True,
                    ),
                ),
            )
        )

        summary = report["trajectories"][0]
        self.assertEqual(summary["fail_to_pass_total"], 2)
        self.assertEqual(summary["fail_to_pass_resolved"], 1)
        self.assertEqual(summary["pass_to_pass_regressed"], 1)
        self.assertFalse(summary["collection_error"])
        self.assertEqual(
            report["graded_failure_attribution"],
            {
                "trials_with_declared_targets": 1,
                "trials_with_any_fail_to_pass_resolved": 1,
                "trials_with_pass_to_pass_regression": 1,
                "trials_with_collection_error": 0,
            },
        )

    def test_trials_without_a_breakdown_report_unknowns_instead_of_zeros(self) -> None:
        report = build_failure_report((_trajectory(),))

        summary = report["trajectories"][0]
        self.assertFalse(summary["graded_targets_declared"])
        self.assertIsNone(summary["fail_to_pass_resolved"])
        self.assertIsNone(summary["pass_to_pass_regressed"])
        self.assertEqual(report["graded_failure_attribution"]["trials_with_declared_targets"], 0)


if __name__ == "__main__":
    unittest.main()
