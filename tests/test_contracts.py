from __future__ import annotations

import unittest

from coding_agent_rl_lab.contracts import RewardVector


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


if __name__ == "__main__":
    unittest.main()
