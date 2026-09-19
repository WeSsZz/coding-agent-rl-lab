from __future__ import annotations

import ast
import re
import tempfile
import unittest
from pathlib import Path

from coding_agent_rl_lab.contracts import ActionKind, AgentAction, TestResult
from coding_agent_rl_lab.evaluation import load_builtin_tasks
from coding_agent_rl_lab.environment import (
    ActionLoopGuard,
    LocalFixtureEnvironment,
    failure_summary,
    render_numbered_window,
    replace_text_mismatch_message,
    search_repository,
    verifier_output_detail,
)


def _write_importing_test_repository(repository: Path) -> None:
    """A repository whose only hit for the searched route is the test that imports the fix."""

    (repository / "tests").mkdir()
    (repository / "pkg" / "core").mkdir(parents=True)
    (repository / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repository / "pkg" / "core" / "__init__.py").write_text("", encoding="utf-8")
    (repository / "pkg" / "core" / "config.py").write_text("BATCH = 1\n", encoding="utf-8")
    (repository / "tests" / "test_config.py").write_text(
        'from pkg.core.config import BATCH\n\n\ndef test_api():\n'
        '    response = get("/moto-api/config")\n',
        encoding="utf-8",
    )


def _write_route_test_repository(repository: Path) -> None:
    """A repository whose route table spells the searched path only as a pattern."""

    (repository / "tests").mkdir()
    (repository / "pkg" / "api").mkdir(parents=True)
    (repository / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repository / "pkg" / "api" / "__init__.py").write_text("", encoding="utf-8")
    (repository / "pkg" / "api" / "urls.py").write_text(
        "url_paths = {\n"
        '    "{0}/moto-api/$": dashboard,\n'
        '    "{0}/moto-api/reset": reset,\n'
        "}\n",
        encoding="utf-8",
    )
    (repository / "tests" / "test_config.py").write_text(
        "from pkg.api.urls import url_paths\n\n\ndef test_api():\n"
        '    response = get("/moto-api/config")\n',
        encoding="utf-8",
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

    def test_finish_without_an_edit_is_refused_while_the_verifier_fails(self) -> None:
        with LocalFixtureEnvironment(self.root) as environment:
            environment.reset(self.task)
            refused = environment.step(AgentAction(ActionKind.FINISH))

            self.assertFalse(refused.terminated)
            self.assertIsNone(refused.violation)
            self.assertTrue(refused.observation.startswith("Tool error: finish refused"))
            self.assertIn("no source file has been edited", refused.observation)
            self.assertIn("Tests failed", refused.observation)

    def test_finish_is_accepted_after_the_source_edit(self) -> None:
        with LocalFixtureEnvironment(self.root) as environment:
            environment.reset(self.task)
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
            finished = environment.step(AgentAction(ActionKind.FINISH))

            self.assertTrue(finished.terminated)
            self.assertTrue(finished.test_result.passed)

    def test_replace_text_quotes_the_exact_text_when_only_whitespace_differs(self) -> None:
        with LocalFixtureEnvironment(self.root) as environment:
            environment.reset(self.task)
            (environment.repository / "sample.py").write_text(
                "def is_fixed():\n    return False\n",
                encoding="utf-8",
            )
            failed = environment.step(
                AgentAction(
                    ActionKind.REPLACE_TEXT,
                    {
                        "path": "sample.py",
                        "old": "def is_fixed(): return False",
                        "new": "def is_fixed():\n    return True",
                    },
                )
            )
            quoted = re.search(r"byte for byte: ('.+?')\.", failed.observation)
            self.assertIsNotNone(quoted, failed.observation)
            exact = ast.literal_eval(quoted.group(1))
            repaired = environment.step(
                AgentAction(
                    ActionKind.REPLACE_TEXT,
                    {"path": "sample.py", "old": exact, "new": "def is_fixed():\n    return True"},
                )
            )

            self.assertFalse(failed.terminated)
            self.assertIn("lines 1-2", failed.observation)
            self.assertEqual(exact, "def is_fixed():\n    return False")
            self.assertEqual(repaired.observation, "Updated sample.py.")

    def test_replace_text_lists_every_ambiguous_match(self) -> None:
        content = "value = 1\nother = 2\nvalue = 1\n"

        message = replace_text_mismatch_message(content, "value = 1")

        self.assertIn("found 2: lines 1, 3", message)
        self.assertIn("matches exactly once", message)

    def test_replace_text_mismatch_without_a_close_match_points_at_a_read(self) -> None:
        message = replace_text_mismatch_message("value = 1\n", "absent_call()")

        self.assertIn("found 0", message)
        self.assertIn("numbered read_file output", message)
        self.assertIn("run search_text with it", message)

    def test_search_points_at_the_implementation_when_every_match_is_a_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            _write_importing_test_repository(repository)

            test_only = search_repository(repository, "/moto-api/config")
            implementation_hit = search_repository(repository, "BATCH")

        self.assertEqual(
            test_only.splitlines(),
            [
                'tests/test_config.py:5:    response = get("/moto-api/config")',
                "IMPLEMENTATION_CANDIDATE:pkg/core/config.py",
            ],
        )
        self.assertNotIn("IMPLEMENTATION_CANDIDATE", implementation_hit)
        self.assertNotIn("Shorter query", implementation_hit)

    def test_search_re_queries_the_path_when_only_the_test_spells_it_out(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            _write_route_test_repository(repository)

            rendered = search_repository(repository, "/moto-api/config")

        self.assertEqual(
            rendered.splitlines(),
            [
                'tests/test_config.py:5:    response = get("/moto-api/config")',
                'No implementation file contains "/moto-api/config". Shorter query "moto-api" '
                "matches implementation files:",
                'pkg/api/urls.py:2:    "{0}/moto-api/$": dashboard,',
                'pkg/api/urls.py:3:    "{0}/moto-api/reset": reset,',
                "IMPLEMENTATION_CANDIDATE:pkg/api/urls.py",
            ],
        )

    def test_recovery_directive_sends_the_edit_to_the_implementation(self) -> None:
        test_only_guard = ActionLoopGuard()
        test_only_guard.record(
            AgentAction(ActionKind.READ_FILE, {"path": "tests/test_config.py"}),
            "1: from pkg.core.config import BATCH",
        )
        test_only_guard.record(
            AgentAction(ActionKind.SEARCH_TEXT, {"query": "/moto-api/config"}),
            "tests/test_config.py:5:    response = get(\"/moto-api/config\")",
        )
        test_only_guard.record(
            AgentAction(ActionKind.FINISH),
            "Tool error: finish refused: the verifier still fails",
        )
        implementation_guard = ActionLoopGuard()
        implementation_guard.record(
            AgentAction(ActionKind.READ_FILE, {"path": "pkg/core/config.py"}),
            "1: BATCH = 1",
        )
        implementation_guard.record(
            AgentAction(ActionKind.FINISH),
            "Tool error: finish refused: the verifier still fails",
        )

        test_only_guard.rejection_for(AgentAction(ActionKind.FINISH))
        test_only_directive = test_only_guard.rejection_for(AgentAction(ActionKind.FINISH))
        implementation_guard.rejection_for(AgentAction(ActionKind.FINISH))
        implementation_directive = implementation_guard.rejection_for(
            AgentAction(ActionKind.FINISH)
        )

        self.assertIn("Only verifier-owned tests have been read", test_only_directive)
        self.assertIn(
            "The edit belongs in the source file that produces the failing value",
            test_only_directive,
        )
        self.assertIn("Search for the exact value the failure quotes", test_only_directive)
        self.assertNotIn("edit a file you already read", test_only_directive.casefold())
        self.assertIn(
            "Implementation files already read: pkg/core/config.py.",
            implementation_directive,
        )

    def test_edit_that_would_break_the_file_is_rejected(self) -> None:
        with LocalFixtureEnvironment(self.root) as environment:
            environment.reset(self.task)
            environment.step(AgentAction(ActionKind.READ_FILE, {"path": "calculator.py"}))
            rejected = environment.step(
                AgentAction(
                    ActionKind.REPLACE_LINES,
                    {
                        "path": "calculator.py",
                        "start_line": 4,
                        "end_line": 4,
                        "new": "        return list(range(start, end + 1))",
                    },
                )
            )

            self.assertTrue(rejected.observation.startswith("Tool error: edit not applied"))
            self.assertIn("line 4", rejected.observation)
            self.assertEqual(environment.changed_files(), ())
            self.assertEqual(
                (environment.repository / "calculator.py").read_text(encoding="utf-8"),
                "def inclusive_range(start: int, end: int) -> list[int]:\n"
                '    """Return every integer from start through end."""\n'
                "\n"
                "    return list(range(start, end))\n",
            )

    def test_verifier_output_detail_keeps_the_stream_that_names_the_failure(self) -> None:
        result = TestResult(
            ("pytest",),
            False,
            4,
            "collected 0 items\nImportError while loading conftest\n"
            "ERROR: found no collectors for tests/test_core/test_config.py::test_api",
            "ERROR: found no collectors for tests/test_core/test_config.py::test_api",
            1.0,
            False,
        )

        detail = verifier_output_detail(result)

        self.assertIn("ImportError while loading conftest", detail)
        self.assertEqual(detail.count("found no collectors"), 1)

    def test_failure_summary_lifts_the_node_error_and_frame_literals(self) -> None:
        result = TestResult(
            ("pytest",),
            False,
            1,
            "tests/test_core/test_config.py F\n"
            "self = <json.decoder.JSONDecoder object at 0x71cf>\n"
            "s = 'Not yet implemented', idx = 0\n"
            ">       assert resp.json()['batch'] == {'use_docker': True}\n"
            "\n"
            "tests/test_core/test_config.py:20: \n"
            ">           raise JSONDecodeError('Expecting value', s)\n"
            "\n"
            "/opt/miniconda3/lib/python3.12/site-packages/requests/models.py:978: \n"
            "E           json.decoder.JSONDecodeError: Expecting value: line 1 column 1\n"
            "FAILED tests/test_core/test_config.py::test_change_configuration_using_api - ...\n",
            "",
            1.0,
            False,
        )

        summary = failure_summary(result)

        self.assertEqual(
            summary.splitlines(),
            [
                "[failing tests] tests/test_core/test_config.py::"
                "test_change_configuration_using_api",
                "[failing statement] tests/test_core/test_config.py:20: "
                "assert resp.json()['batch'] == {'use_docker': True}",
                "[last error] json.decoder.JSONDecodeError: Expecting value: line 1 column 1",
                "[string values in the failing frame] s = 'Not yet implemented'",
            ],
        )
        self.assertEqual(
            failure_summary(TestResult(("pytest",), True, 0, "1 passed", "", 1.0, False)),
            "",
        )

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
