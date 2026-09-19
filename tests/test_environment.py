from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from coding_agent_rl_lab.contracts import ActionKind, AgentAction
from coding_agent_rl_lab.evaluation import load_builtin_tasks
from coding_agent_rl_lab.environment import (
    LocalFixtureEnvironment,
    render_numbered_window,
    search_repository,
)


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

    def test_replace_lines_updates_a_previously_read_source_range(self) -> None:
        with LocalFixtureEnvironment(self.root) as environment:
            environment.reset(self.task)
            environment.step(AgentAction(ActionKind.READ_FILE, {"path": "calculator.py"}))
            replaced = environment.step(
                AgentAction(
                    ActionKind.REPLACE_LINES,
                    {
                        "path": "calculator.py",
                        "start_line": 4,
                        "end_line": 4,
                        "new": "    return list(range(start, end + 1))",
                    },
                )
            )
            result = environment.step(AgentAction(ActionKind.RUN_TESTS))

            self.assertEqual(replaced.observation, "Updated calculator.py.")
            self.assertTrue(result.test_result.passed)
            self.assertEqual(environment.changed_files(), ("calculator.py",))

    def test_replace_lines_requires_a_fresh_read(self) -> None:
        with LocalFixtureEnvironment(self.root) as environment:
            environment.reset(self.task)
            action = AgentAction(
                ActionKind.REPLACE_LINES,
                {"path": "calculator.py", "start_line": 4, "end_line": 4, "new": "pass"},
            )

            unread = environment.step(action)
            environment.step(AgentAction(ActionKind.READ_FILE, {"path": "calculator.py"}))
            environment.step(action)
            stale = environment.step(action)

            self.assertIn("requires reading the target file first", unread.observation)
            self.assertIn("requires reading the target file first", stale.observation)

    def test_path_escape_is_a_hard_violation(self) -> None:
        with LocalFixtureEnvironment(self.root) as environment:
            environment.reset(self.task)
            result = environment.step(AgentAction(ActionKind.READ_FILE, {"path": "../../etc/passwd"}))
            self.assertTrue(result.terminated)
            self.assertIsNotNone(result.violation)
            self.assertEqual(
                environment.violations,
                ["invalid_action:read_file:path_escape"],
            )

    def test_missing_file_is_a_recoverable_tool_error(self) -> None:
        with LocalFixtureEnvironment(self.root) as environment:
            environment.reset(self.task)
            result = environment.step(AgentAction(ActionKind.READ_FILE, {"path": "missing.py"}))
            self.assertFalse(result.terminated)
            self.assertIsNone(result.violation)
            self.assertIn("Tool error: file does not exist", result.observation)
            self.assertEqual(environment.violations, [])

    def test_missing_tool_argument_is_recoverable(self) -> None:
        with LocalFixtureEnvironment(self.root) as environment:
            environment.reset(self.task)
            result = environment.step(AgentAction(ActionKind.SEARCH_TEXT))

            self.assertFalse(result.terminated)
            self.assertIsNone(result.violation)
            self.assertIn("Tool error: query must be a non-empty string", result.observation)
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

    def test_read_file_numbers_lines_so_edits_do_not_have_to_recount(self) -> None:
        with LocalFixtureEnvironment(self.root) as environment:
            environment.reset(self.task)
            result = environment.step(
                AgentAction(
                    ActionKind.READ_FILE,
                    {"path": "calculator.py", "start_line": 1, "end_line": 20},
                )
            )

            numbered = result.observation.splitlines()[0]
            self.assertEqual(numbered.split(": ", 1)[0], "1")
            self.assertNotIn("[read_file lines", result.observation)

    def test_escalating_rejections_tell_the_policy_what_it_already_did(self) -> None:
        with LocalFixtureEnvironment(self.root) as environment:
            environment.reset(self.task)
            action = AgentAction(ActionKind.SEARCH_TEXT, {"query": "range(start, end)"})
            environment.step(action)
            first = environment.step(action)
            second = environment.step(action)

            self.assertIn("Tool error: do not repeat a search_text query", first.observation)
            self.assertNotIn("Repeated rejection", first.observation)
            self.assertIn("Repeated rejection", second.observation)
            self.assertIn("No patch has been applied yet.", second.observation)
            self.assertIn("Queries already used: range(start, end).", second.observation)
            self.assertEqual(environment.loop_rejections, 2)


class RenderNumberedWindowTests(unittest.TestCase):
    def test_empty_file_is_explicit_instead_of_blank(self) -> None:
        self.assertEqual(render_numbered_window(""), "[file is empty]")

    def test_truncated_window_names_the_next_start_line(self) -> None:
        content = "\n".join(f"line {number}" for number in range(1, 11))

        rendered = render_numbered_window(content, None, max_lines=4)

        self.assertEqual(
            rendered.splitlines()[-1],
            "[read_file lines 1-4: file has 10 lines; "
            "continue with read_file start_line=5 end_line=10]",
        )

    def test_character_budget_still_numbers_every_returned_line(self) -> None:
        rendered = render_numbered_window("alpha\nbeta\ngamma", max_chars=1)

        self.assertEqual(rendered.splitlines()[0], "1: alpha")
        self.assertIn("character budget reached", rendered)


class SearchRepositoryTests(unittest.TestCase):
    def test_ranked_matches_are_capped_per_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("docs/guide.md", "tests/test_feature.py", "src/feature.py"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("needle\n", encoding="utf-8")
            (root / "src" / "noisy.py").write_text("needle\n" * 20, encoding="utf-8")

            rendered = search_repository(root, "needle")

        matches = rendered.splitlines()
        self.assertEqual(matches[0], "src/feature.py:1:needle")
        self.assertEqual(sum(line.startswith("src/noisy.py:") for line in matches), 5)
        self.assertEqual(
            sorted({line.split(":")[0] for line in matches}),
            ["docs/guide.md", "src/feature.py", "src/noisy.py", "tests/test_feature.py"],
        )

    def test_failed_query_is_retried_with_its_longest_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src" / "client.py").write_text(
                'CALLBACK = "waitForTaskToken"\n',
                encoding="utf-8",
            )

            rendered = search_repository(root, "how should a callback wait")

        self.assertEqual(
            rendered.splitlines(),
            [
                "No exact matches for: how should a callback wait",
                "Longest token in the query: callback",
                "src/client.py:1:" + 'CALLBACK = "waitForTaskToken"',
            ],
        )

    def test_unmatched_query_without_a_long_token_suggests_a_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "moto").mkdir()
            (root / "moto" / "models.py").write_text("value = 1\n", encoding="utf-8")

            rendered = search_repository(root, "moto/models/__init__.py")

        self.assertIn("No exact matches for: moto/models/__init__.py", rendered)
        self.assertIn("SUGGESTED_PATH:moto/models.py", rendered)

    def test_long_match_lists_keep_the_top_matches_and_say_what_was_cut(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src" / "noisy.py").write_text(
                "\n".join(f"needle {'x' * 100}" for _ in range(200)),
                encoding="utf-8",
            )
            (root / "src" / "first.py").write_text("needle\n", encoding="utf-8")

            rendered = search_repository(root, "needle", per_file_limit=200)

        lines = rendered.splitlines()
        self.assertEqual(lines[0], "src/first.py:1:needle")
        noisy_matches = sum(line.startswith("src/noisy.py:") for line in lines)
        self.assertGreater(noisy_matches, 0)
        self.assertLess(noisy_matches, 200)
        self.assertRegex(
            lines[-1],
            r"^\[search_text: showing \d+ of \d+ matches; "
            r"narrow the query or read the first file\]$",
        )
        self.assertLess(len(rendered), 8_300)


if __name__ == "__main__":
    unittest.main()
