from __future__ import annotations

import unittest
from dataclasses import replace

from coding_agent_rl_lab.contracts import TestResult
from coding_agent_rl_lab.reward_shaping import build_training_reward, verifier_breakdown


def _result(*failed: str, passed: bool = False, timed_out: bool = False) -> TestResult:
    summary = "\n".join(f"FAILED {item} - assertion" for item in failed)
    if failed:
        summary += f"\n{len(failed)} failed"
    return TestResult(
        command=("pytest",),
        passed=passed,
        exit_code=0 if passed else 1,
        stdout=summary,
        stderr="",
        duration_ms=1.0,
        timed_out=timed_out,
    )


class TrainingRewardTests(unittest.TestCase):
    def test_conservative_reward_distinguishes_progress_from_safe_editing(self):
        baseline = _result("a", "b")
        for final, expected in ((baseline, 0.03), (_result("b"), 0.255), (_result(passed=True), 1.0)):
            with self.subTest(expected=expected):
                reward = build_training_reward(
                    baseline=baseline, final=final, patch_created=True, patch_valid=True,
                    verifier_run_after_patch=True, violations=(), reward_version="conservative-v2",
                )
                self.assertEqual(reward.training_reward, expected)
                self.assertEqual(reward.to_dict()["reward_version"], "conservative-v2")

    def test_conservative_reward_rejects_new_failures_and_unknown_verifier_status(self):
        baseline = _result("a", "b", "c")
        finals = [None, _result(), _result("b", "new"),
                  _result("b", timed_out=True), replace(_result("b"), exit_code=2),
                  replace(_result("b"), stderr="ERROR collecting tests/test_other.py"),
                  replace(_result("b"), stderr="1 error")]
        for final in finals:
            with self.subTest(final=final):
                reward = build_training_reward(
                    baseline=baseline, final=final, patch_created=True, patch_valid=True,
                    verifier_run_after_patch=True, violations=(), reward_version="conservative-v2",
                )
                self.assertEqual(reward.training_reward, 0.0)

    def test_conservative_progress_requires_valid_verified_patch(self):
        for changes in ({"patch_valid": False}, {"verifier_run_after_patch": False}, {"patch_created": False}):
            args = dict(baseline=_result("a", "b"), final=_result("b"), patch_created=True,
                        patch_valid=True, verifier_run_after_patch=True, violations=(), reward_version="conservative-v2")
            reward = build_training_reward(**{**args, **changes})
            self.assertEqual(reward.training_reward, 0.0)

    def test_conservative_violation_overrides_success(self):
        reward = build_training_reward(
            baseline=_result("a"), final=_result(passed=True), patch_created=True, patch_valid=True,
            verifier_run_after_patch=True, violations=("test_tampering",), reward_version="conservative-v2",
        )
        self.assertEqual(reward.training_reward, -1.0)
        self.assertEqual(reward.strict_reward, 0.0)

    def test_unknown_final_count_does_not_claim_resolved_or_no_new_failures(self) -> None:
        for final in (None, _result()):
            with self.subTest(final=final):
                reward = build_training_reward(
                    baseline=_result("tests/test_a.py::test_a"), final=final,
                    patch_created=True, patch_valid=False,
                    verifier_run_after_patch=True, violations=(),
                )
                self.assertIsNone(reward.final_failure_count)
                self.assertEqual(reward.resolved_failure_count, 0)
                self.assertIsNone(reward.new_failure_count)
                self.assertEqual(reward.training_reward, 0.0)

    def test_strict_success_is_one(self) -> None:
        reward = build_training_reward(
            baseline=_result("tests/test_a.py::test_a"),
            final=_result(passed=True),
            patch_created=True,
            patch_valid=True,
            verifier_run_after_patch=True,
            violations=(),
        )

        self.assertEqual(reward.strict_reward, 1.0)
        self.assertEqual(reward.training_reward, 1.0)

    def test_reducing_failures_earns_bounded_partial_reward(self) -> None:
        reward = build_training_reward(
            baseline=_result("tests/test_a.py::test_a", "tests/test_b.py::test_b"),
            final=_result("tests/test_b.py::test_b"),
            patch_created=True,
            patch_valid=True,
            verifier_run_after_patch=True,
            violations=(),
        )

        self.assertEqual(reward.strict_reward, 0.0)
        self.assertEqual(reward.resolved_failure_count, 1)
        self.assertEqual(reward.new_failure_count, 0)
        self.assertEqual(reward.training_reward, 0.375)

    def test_valid_patch_without_test_progress_cannot_exceed_small_bonus(self) -> None:
        baseline = _result("tests/test_a.py::test_a")
        reward = build_training_reward(
            baseline=baseline,
            final=baseline,
            patch_created=True,
            patch_valid=True,
            verifier_run_after_patch=True,
            violations=(),
        )

        self.assertEqual(reward.training_reward, 0.15)

    def test_patch_without_post_edit_verifier_gets_no_reward(self) -> None:
        reward = build_training_reward(
            baseline=_result("tests/test_a.py::test_a"),
            final=_result("tests/test_a.py::test_a"),
            patch_created=True,
            patch_valid=True,
            verifier_run_after_patch=False,
            violations=(),
        )

        self.assertEqual(reward.training_reward, 0.0)

    def test_new_failure_blocks_regression_free_bonus(self) -> None:
        reward = build_training_reward(
            baseline=_result("tests/test_a.py::test_a", "tests/test_b.py::test_b"),
            final=_result("tests/test_b.py::test_b", "tests/test_c.py::test_c"),
            patch_created=True,
            patch_valid=True,
            verifier_run_after_patch=True,
            violations=(),
        )

        self.assertEqual(reward.new_failure_count, 1)
        self.assertEqual(reward.training_reward, 0.15)

    def test_violation_is_negative_even_if_tests_pass(self) -> None:
        reward = build_training_reward(
            baseline=_result("tests/test_a.py::test_a"),
            final=_result(passed=True),
            patch_created=True,
            patch_valid=True,
            verifier_run_after_patch=True,
            violations=("invalid_action:EnvironmentError",),
        )

        self.assertEqual(reward.strict_reward, 0.0)
        self.assertEqual(reward.training_reward, -1.0)

    def test_timeout_gets_no_partial_reward(self) -> None:
        reward = build_training_reward(
            baseline=_result("tests/test_a.py::test_a"),
            final=_result("tests/test_a.py::test_a", timed_out=True),
            patch_created=True,
            patch_valid=True,
            verifier_run_after_patch=True,
            violations=(),
        )

        self.assertEqual(reward.training_reward, 0.0)


class VerifierBreakdownTests(unittest.TestCase):
    def test_declared_nodes_are_attributed_without_claiming_a_fix(self) -> None:
        breakdown = verifier_breakdown(
            _result("tests/test_bug.py::test_decimal"),
            fail_to_pass=("tests/test_bug.py::test_decimal",),
            pass_to_pass=("tests/test_existing.py::test_regression",),
        )

        self.assertTrue(breakdown.comparable)
        self.assertEqual(breakdown.fail_to_pass_total, 1)
        self.assertEqual(breakdown.fail_to_pass_resolved, 0)
        self.assertEqual(breakdown.pass_to_pass_total, 1)
        self.assertEqual(breakdown.pass_to_pass_regressed, 0)
        self.assertEqual(breakdown.ungraded_failed_nodes, ())
        self.assertTrue(breakdown.regression_free)

    def test_regressions_and_ungraded_failures_are_separated(self) -> None:
        breakdown = verifier_breakdown(
            _result(
                "tests/test_existing.py::test_regression",
                "tests/test_other.py::test_unrelated",
            ),
            fail_to_pass=("tests/test_bug.py::test_decimal",),
            pass_to_pass=("tests/test_existing.py::test_regression",),
        )

        self.assertEqual(breakdown.pass_to_pass_regressed, 1)
        self.assertEqual(breakdown.failed_nodes, (
            "tests/test_existing.py::test_regression",
            "tests/test_other.py::test_unrelated",
        ))
        self.assertEqual(breakdown.ungraded_failed_nodes, ("tests/test_other.py::test_unrelated",))
        self.assertFalse(breakdown.regression_free)

    def test_parameterized_node_ids_count_as_their_declared_target(self) -> None:
        breakdown = verifier_breakdown(
            _result("tests/test_bug.py::test_decimal[1.5]"),
            fail_to_pass=("tests/test_bug.py::test_decimal",),
        )

        self.assertEqual(breakdown.fail_to_pass_resolved, 0)
        self.assertEqual(breakdown.ungraded_failed_nodes, ())

    def test_passing_run_resolves_every_declared_target(self) -> None:
        breakdown = verifier_breakdown(
            _result(passed=True),
            fail_to_pass=("tests/test_bug.py::test_decimal",),
            pass_to_pass=("tests/test_existing.py::test_regression",),
        )

        self.assertEqual(breakdown.fail_to_pass_resolved, 1)
        self.assertEqual(breakdown.pass_to_pass_regressed, 0)
        self.assertTrue(breakdown.regression_free)

    def test_collection_error_and_missing_summary_report_unknown_not_zero(self) -> None:
        collection = verifier_breakdown(
            replace(_result(), stderr="ERROR collecting tests/test_bug.py"),
            fail_to_pass=("tests/test_bug.py::test_decimal",),
        )
        silent = verifier_breakdown(
            _result(),
            fail_to_pass=("tests/test_bug.py::test_decimal",),
        )

        self.assertTrue(collection.collection_error)
        self.assertFalse(collection.comparable)
        self.assertIsNone(collection.fail_to_pass_resolved)
        self.assertFalse(collection.regression_free)
        self.assertIsNone(silent.fail_to_pass_resolved)
        self.assertIsNone(silent.pass_to_pass_regressed)
        self.assertFalse(silent.regression_free)

    def test_undeclared_runs_keep_the_failure_list_without_comparability(self) -> None:
        breakdown = verifier_breakdown(_result("tests/test_a.py::test_a"))

        self.assertFalse(breakdown.comparable)
        self.assertFalse(breakdown.node_targets_declared)
        self.assertEqual(breakdown.failed_nodes, ("tests/test_a.py::test_a",))
        self.assertIsNone(verifier_breakdown(None))

    def test_training_reward_carries_the_declared_breakdown(self) -> None:
        reward = build_training_reward(
            baseline=_result("tests/test_bug.py::test_decimal"),
            final=_result("tests/test_bug.py::test_decimal"),
            patch_created=True,
            patch_valid=True,
            verifier_run_after_patch=True,
            violations=(),
            reward_version="conservative-v2",
            fail_to_pass=("tests/test_bug.py::test_decimal",),
            pass_to_pass=("tests/test_existing.py::test_regression",),
        )

        self.assertEqual(reward.verifier.fail_to_pass_resolved, 0)
        self.assertEqual(reward.verifier.pass_to_pass_regressed, 0)
        self.assertEqual(reward.to_dict()["verifier"]["pass_to_pass_total"], 1)
        self.assertEqual(reward.to_dict()["verifier"]["collection_error"], False)


if __name__ == "__main__":
    unittest.main()
