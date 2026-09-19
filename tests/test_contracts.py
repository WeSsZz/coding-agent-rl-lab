from __future__ import annotations

import json
import unittest

from coding_agent_rl_lab.contracts import RewardVector, VerifierBreakdown


class RewardVectorTests(unittest.TestCase):
    def test_hard_violation_zeros_scalar_reward(self) -> None:
        reward = RewardVector(
            task_success=True,
            tests_passed=True,
            regression_free=True,
            patch_created=True,
            tool_calls=2,
            steps=2,
            violations=("path_escape",),
        )
        self.assertEqual(reward.scalar, 0.0)

    def test_successful_efficient_patch_receives_positive_reward(self) -> None:
        reward = RewardVector(True, True, True, True, 4, 4)
        self.assertGreaterEqual(reward.scalar, 0.9)

    def test_failed_task_does_not_receive_efficiency_bonus(self) -> None:
        reward = RewardVector(False, False, False, False, 1, 1)
        self.assertEqual(reward.scalar, 0.0)


class VerifierBreakdownTests(unittest.TestCase):
    def test_unknown_counts_never_claim_regression_free(self) -> None:
        unknown = VerifierBreakdown(node_targets_declared=True)
        collection_error = VerifierBreakdown(node_targets_declared=True, collection_error=True)

        self.assertTrue(unknown.comparable)
        self.assertIsNone(unknown.pass_to_pass_regressed)
        self.assertFalse(unknown.regression_free)
        self.assertFalse(collection_error.comparable)
        self.assertFalse(collection_error.regression_free)

    def test_ungraded_failures_block_regression_free(self) -> None:
        breakdown = VerifierBreakdown(
            node_targets_declared=True,
            pass_to_pass_regressed=0,
            ungraded_failed_nodes=("tests/test_other.py::test_unrelated",),
        )

        self.assertTrue(breakdown.comparable)
        self.assertFalse(breakdown.regression_free)

    def test_json_round_trip_preserves_the_breakdown(self) -> None:
        breakdown = VerifierBreakdown(
            fail_to_pass_total=2,
            pass_to_pass_total=1,
            fail_to_pass_resolved=1,
            pass_to_pass_regressed=0,
            failed_nodes=("tests/test_bug.py::test_decimal",),
            node_targets_declared=True,
        )

        reloaded = VerifierBreakdown.from_dict(json.loads(json.dumps(breakdown.to_dict())))

        self.assertEqual(reloaded, breakdown)
        self.assertIsNone(VerifierBreakdown.from_dict({"pass_to_pass_regressed": None}).pass_to_pass_regressed)


if __name__ == "__main__":
    unittest.main()
