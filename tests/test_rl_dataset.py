from __future__ import annotations

import unittest

from coding_agent_rl_lab.rl_dataset import export_rl_episodes


def _trajectory(*, violations=(), success=False, contains_answers=False):
    return {
        "schema_version": 3,
        "trajectory_id": "traj-1",
        "task_id": "task-1",
        "repetition": 1,
        "seed": 7,
        "policy": {
            "policy_id": "model",
            "version": "1",
            "policy_type": "model",
            "model": "coder",
            "metadata": {"contains_answers": contains_answers},
        },
        "steps": [
            {
                "sequence": 1,
                "action": {"kind": "finish", "arguments": {}},
                "observation": "tests failed",
                "policy_input": [{"role": "user", "content": "task"}],
                "policy_output": '{"kind":"finish","arguments":{}}',
                "policy_metadata": {"usage": {"total_tokens": 10}},
                "violation": violations[0] if violations else None,
            }
        ],
        "reward": {
            "task_success": success,
            "tests_passed": success,
            "regression_free": success,
            "patch_created": success,
            "tool_calls": 1,
            "steps": 1,
            "violations": list(violations),
        },
        "changed_files": ["code.py"] if success else [],
        "baseline_tests_passed": False,
        "final_tests_passed": success,
        "initial_observation": "baseline failed",
    }


class RLDatasetTests(unittest.TestCase):
    def test_valid_model_failure_is_kept_as_zero_reward_episode(self) -> None:
        episodes, report = export_rl_episodes(
            (_trajectory(violations=("policy_protocol_error",)),)
        )

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["reward"]["scalar"], 0.0)
        self.assertFalse(episodes[0]["eligible_for_sft"])
        self.assertEqual(report["negative_count"], 1)

    def test_transport_failures_and_answer_policies_are_excluded(self) -> None:
        transport = _trajectory(violations=("policy_transport_error",))
        answers = _trajectory(contains_answers=True)
        answers["trajectory_id"] = "traj-2"

        episodes, report = export_rl_episodes((transport, answers))

        self.assertEqual(episodes, ())
        self.assertEqual(report["excluded_count"], 2)
        self.assertEqual(report["exclusion_reasons"]["model_transport_failure"], 1)
        self.assertEqual(report["exclusion_reasons"]["answer_containing_policy"], 1)

    def test_successful_answer_free_episode_is_sft_eligible(self) -> None:
        episodes, report = export_rl_episodes((_trajectory(success=True),))

        self.assertTrue(episodes[0]["eligible_for_sft"])
        self.assertGreater(episodes[0]["reward"]["scalar"], 0.0)
        self.assertEqual(report["positive_count"], 1)

    def test_sensitive_metadata_is_excluded(self) -> None:
        row = _trajectory()
        row["steps"][0]["policy_metadata"]["authorization"] = "Bearer secret"

        episodes, report = export_rl_episodes((row,))

        self.assertEqual(episodes, ())
        self.assertEqual(report["exclusion_reasons"], {"sensitive_metadata": 1})


if __name__ == "__main__":
    unittest.main()
