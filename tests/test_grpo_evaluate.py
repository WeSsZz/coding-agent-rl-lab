import unittest

from coding_agent_rl_lab.grpo_evaluate import select_rows, summarize, validate_resume
from coding_agent_rl_lab.swe_gym_smoke import pinned_rows_for_task_set


class DirectionEvaluationTests(unittest.TestCase):
    def test_resume_rejects_changed_adapter_budget_and_unexpected_tasks(self):
        manifest = {"adapter_path": "/adapter", "adapter_sha256": "abc", "model_path": "/model",
                    "prompt_rows_sha256": "def", "seed": 81000,
                    "budget": {"num_generations": 2}, "planned_trial_count": 16,
                    "tasks": [{"task_id": "train-task"}]}
        validate_resume(manifest, manifest, {"train-task"})
        for key, value in (("adapter_sha256", "changed"), ("budget", {"num_generations": 4}),
                           ("tasks", [{"task_id": "held-out-task"}]),
                           ("tasks", [{"task_id": "train-task"}] * 2)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_resume({**manifest, key: value}, manifest, {"train-task"})

    def test_split_is_fixed_and_held_out_is_rejected(self):
        rows = [{"task_id": item.instance_id} for item in pinned_rows_for_task_set("all")]
        self.assertEqual(len(select_rows(rows, "train")), 6)
        self.assertEqual(len(select_rows(rows, "regression")), 2)
        with self.assertRaises(ValueError):
            select_rows(rows, "held-out")
        with self.assertRaises(KeyError):
            select_rows(rows[:6], "regression")
        with self.assertRaises(ValueError):
            select_rows(rows + rows[:1], "train")

    def test_edit_reward_does_not_count_as_success_or_resolved_failure(self):
        record = {"reward": 0.15, "reward_components": {
            "strict_success": False, "patch_created": True, "patch_valid": True,
            "verifier_run_after_patch": True, "resolved_failure_count": 0,
            "baseline_failure_count": 3, "final_failure_count": 3,
            "new_failure_count": 0, "violations": [],
        }}
        result = summarize([record])
        self.assertEqual(result["strict_successes"], 0)
        self.assertEqual(result["trials_resolving_failures"], 0)
        self.assertEqual(result["verified_patches"], 1)
        with self.assertRaises(ValueError):
            summarize([{"reward": 0}])

    def test_unknown_final_count_is_not_test_improvement(self):
        record = {"reward": 0, "reward_components": {
            "strict_success": False, "patch_created": True, "patch_valid": False,
            "verifier_run_after_patch": True, "resolved_failure_count": 1,
            "baseline_failure_count": 1, "final_failure_count": None,
            "new_failure_count": 0, "violations": [],
        }}
        result = summarize([record])
        self.assertEqual(result["trials_resolving_failures"], 0)
        self.assertEqual(result["trials_with_unknown_final_failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
