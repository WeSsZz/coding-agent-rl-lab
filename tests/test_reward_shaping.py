from __future__ import annotations

import unittest

from coding_agent_rl_lab.contracts import TestResult
from coding_agent_rl_lab.reward_shaping import build_training_reward


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


if __name__ == "__main__":
    unittest.main()
