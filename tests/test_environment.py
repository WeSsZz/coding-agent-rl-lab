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

    def test_missing_file_is_a_recoverable_tool_error(self) -> None:
        with LocalFixtureEnvironment(self.root) as environment:
            environment.reset(self.task)
            result = environment.step(AgentAction(ActionKind.READ_FILE, {"path": "missing.py"}))
            self.assertFalse(result.terminated)
            self.assertIsNone(result.violation)
            self.assertIn("Tool error: file does not exist", result.observation)
            self.assertEqual(environment.violations, [])

    def test_search_text_finds_literal_content(self) -> None:
        with LocalFixtureEnvironment(self.root) as environment:
            environment.reset(self.task)
            result = environment.step(AgentAction(ActionKind.SEARCH_TEXT, {"query": "range(start, end)"}))
            self.assertFalse(result.terminated)
            self.assertIsNone(result.violation)
            self.assertIn("calculator.py:", result.observation)

    def test_read_file_can_select_a_contextual_line_range(self) -> None:
        with LocalFixtureEnvironment(self.root) as environment:
            environment.reset(self.task)
            result = environment.step(
                AgentAction(
                    ActionKind.READ_FILE,
                    {"path": "calculator.py", "start_line": 1, "end_line": 20},
                )
            )

            self.assertFalse(result.terminated)
            self.assertIsNone(result.violation)
            self.assertIn("inclusive_range", result.observation)

    def test_read_file_rejects_an_invalid_line_range_as_a_tool_error(self) -> None:
        with LocalFixtureEnvironment(self.root) as environment:
            environment.reset(self.task)
            result = environment.step(
                AgentAction(
                    ActionKind.READ_FILE,
                    {"path": "calculator.py", "start_line": 3, "end_line": 2},
                )
            )

            self.assertFalse(result.terminated)
            self.assertIsNone(result.violation)
            self.assertIn("Tool error: read_file requires", result.observation)

    def test_read_file_rejects_too_little_context_as_a_tool_error(self) -> None:
        with LocalFixtureEnvironment(self.root) as environment:
            environment.reset(self.task)
            result = environment.step(
                AgentAction(
                    ActionKind.READ_FILE,
                    {"path": "calculator.py", "start_line": 1, "end_line": 10},
                )
            )

            self.assertFalse(result.terminated)
            self.assertIsNone(result.violation)
            self.assertIn("must include at least 20 lines", result.observation)

    def test_repeated_search_is_a_recoverable_environment_observation(self) -> None:
        with LocalFixtureEnvironment(self.root) as environment:
            environment.reset(self.task)
            action = AgentAction(ActionKind.SEARCH_TEXT, {"query": "range(start, end)"})
            environment.step(action)
            result = environment.step(action)

            self.assertFalse(result.terminated)
            self.assertIsNone(result.violation)
            self.assertIn("Tool error: do not repeat a search_text query", result.observation)
            self.assertEqual(environment.violations, [])


if __name__ == "__main__":
    unittest.main()
