from __future__ import annotations

import unittest
from pathlib import Path

from coding_agent_rl_lab.evaluation import load_builtin_tasks
from coding_agent_rl_lab.grpo_environment import (
    GRPOEnvironmentError,
    build_grpo_environment_factory,
    build_grpo_prompt_rows,
    grpo_verifier_reward,
)
from coding_agent_rl_lab.providers import LocalFixtureEnvironmentProvider


class GRPOEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.tasks = load_builtin_tasks(self.root)
        self.factory = build_grpo_environment_factory(
            self.tasks,
            LocalFixtureEnvironmentProvider(self.root),
        )

    def test_prompt_rows_contain_public_task_data_without_answers(self) -> None:
        rows = build_grpo_prompt_rows(self.tasks)

        self.assertEqual(len(rows), len(self.tasks))
        self.assertEqual(rows[0]["task_id"], self.tasks[0].task_id)
        prompt_text = str(rows[0]["prompt"])
        self.assertIn(self.tasks[0].issue, prompt_text)
        self.assertNotIn("reference_actions", prompt_text)
        self.assertIn("<tool_call>", prompt_text)
        self.assertIn("Never use Markdown fences", prompt_text)

    def test_environment_tools_apply_patch_and_reward_final_state(self) -> None:
        environment = self.factory()
        task = next(task for task in self.tasks if task.task_id == "clamp-negative-values")

        initial = environment.reset(task_id=task.task_id)
        self.assertIn("Tests failed", initial)
        source = environment.read_file("values.py")
        self.assertIn("return value", source)
        environment.replace_text(
            "values.py",
            "    return value\n",
            "    return max(0, value)\n",
        )
        final = environment.finish()

        self.assertIn("OK", final)
        self.assertEqual(grpo_verifier_reward((environment,)), [1.0])
        self.assertEqual(environment.get_reward(), 1.0)

    def test_unknown_task_and_actions_before_reset_are_rejected(self) -> None:
        environment = self.factory()

        self.assertEqual(environment.reward, 0.0)

        with self.assertRaisesRegex(GRPOEnvironmentError, "known task_id"):
            environment.reset(task_id="missing")
        with self.assertRaisesRegex(GRPOEnvironmentError, "not active"):
            environment.list_files()


if __name__ == "__main__":
    unittest.main()
