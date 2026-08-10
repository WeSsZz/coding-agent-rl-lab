from __future__ import annotations

import unittest
from pathlib import Path

from coding_agent_rl_lab.contracts import ActionKind, AgentAction
from coding_agent_rl_lab.evaluation import load_builtin_tasks
from coding_agent_rl_lab.environment import LocalFixtureEnvironment


class EnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.task = load_builtin_tasks(self.root)[0]

    def test_baseline_fails_and_reference_patch_passes(self) -> None:
        with LocalFixtureEnvironment(self.root) as environment:
            environment.reset(self.task)
            self.assertFalse(environment.baseline_result.passed)
            environment.step(
                AgentAction(
                    ActionKind.REPLACE_TEXT,
                    {
                        "path": "calculator.py",
                        "old": "return list(range(start, end))",
                        "new": "return list(range(start, end + 1))",
                    },
                )
            )
            result = environment.step(AgentAction(ActionKind.RUN_TESTS))
            self.assertTrue(result.terminated)
            self.assertTrue(result.test_result.passed)
            self.assertEqual(environment.changed_files(), ("calculator.py",))

    def test_path_escape_is_a_hard_violation(self) -> None:
        with LocalFixtureEnvironment(self.root) as environment:
            environment.reset(self.task)
            result = environment.step(AgentAction(ActionKind.READ_FILE, {"path": "../../etc/passwd"}))
            self.assertTrue(result.terminated)
            self.assertIsNotNone(result.violation)
            self.assertEqual(environment.violations, ["invalid_action:EnvironmentError"])


if __name__ == "__main__":
    unittest.main()
