from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from coding_agent_rl_lab.contracts import ActionKind, AgentAction, CodingTask, DatasetSplit
from coding_agent_rl_lab.docker_environment import (
    CommandExecution,
    DockerSandboxConfig,
    DockerSandboxEnvironment,
    DockerTaskSpec,
)
from coding_agent_rl_lab.environment import (
    python_edit_syntax_error,
    replace_text_mismatch_message,
    search_repository,
)


class FakeDockerRunner:
    def __init__(self, base_commit: str) -> None:
        self.base_commit = base_commit
        self.calls: list[tuple[tuple[str, ...], str | None, float | None]] = []
        self.test_runs = 0
        self.timed_out_test_runs: set[int] = set()

    def run(
        self,
        argv: tuple[str, ...],
        *,
        input_text: str | None = None,
        timeout_seconds: float | None = None,
    ) -> CommandExecution:
        self.calls.append((argv, input_text, timeout_seconds))
        if argv[:2] == ("docker", "run"):
            return CommandExecution(0, stdout="container-id")
        if argv[:3] == ("docker", "rm", "--force"):
            return CommandExecution(0)
        if argv[-3:] == ("git", "rev-parse", "HEAD"):
            return CommandExecution(0, stdout=self.base_commit + "\n")
        if argv[-4:] == ("git", "apply", "--whitespace=nowarn", "-"):
            return CommandExecution(0)
        if argv[-4:] == ("python", "-m", "pytest", "-q"):
            self.test_runs += 1
            if self.test_runs in self.timed_out_test_runs:
                return CommandExecution(124, stderr="timed out", duration_ms=30_000.0, timed_out=True)
            if self.test_runs == 1:
                return CommandExecution(1, stdout="1 failed", duration_ms=12.0)
            return CommandExecution(0, stdout="1 passed", duration_ms=10.0)
        if argv[-1:] == ("missing.py",):
            return CommandExecution(1, stderr="FileNotFoundError: missing.py")
        if argv[-1:] == ("needle",):
            return CommandExecution(0, stdout="src/bug.py:7:needle\n")
        if "replace_text requires exactly one match" in " ".join(argv):
            return CommandExecution(0)
        return CommandExecution(0)


class _AlwaysFailingDockerRunner(FakeDockerRunner):
    """A container whose tests never pass, so an applied patch can still be the wrong fix."""

    def run(
        self,
        argv: tuple[str, ...],
        *,
        input_text: str | None = None,
        timeout_seconds: float | None = None,
    ) -> CommandExecution:
        execution = super().run(argv, input_text=input_text, timeout_seconds=timeout_seconds)
        if argv[-4:] == ("python", "-m", "pytest", "-q"):
            return CommandExecution(1, stdout="1 failed", duration_ms=11.0)
        return execution


class DockerEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_commit = "a" * 40
        self.task = CodingTask(
            task_id="owner__repo-1",
            issue="Fix the bug.",
            fixture_path=None,
            base_commit=self.base_commit,
            test_command=("python", "-m", "pytest", "-q"),
            split=DatasetSplit.DEVELOPMENT,
            provenance="test",
            max_steps=5,
        )
        self.spec = DockerTaskSpec(
            task_id=self.task.task_id,
            image="example/task:latest",
            base_commit=self.base_commit,
            test_patch=(
                "diff --git a/tests/test_bug.py b/tests/test_bug.py\n"
                "--- a/tests/test_bug.py\n"
                "+++ b/tests/test_bug.py\n"
            ),
            fail_to_pass=("tests/test_bug.py::test_bug",),
        )

    def test_fail_before_replace_and_pass_after_lifecycle(self) -> None:
        runner = FakeDockerRunner(self.base_commit)
        environment = DockerSandboxEnvironment(self.spec, DockerSandboxConfig(), runner)
        try:
            observation = environment.reset(self.task)
            self.assertIn("Baseline verifier result", observation)
            self.assertIn("Tests failed", observation)
            self.assertFalse(environment.baseline_result.passed)

            replaced = environment.step(
                AgentAction(
                    ActionKind.REPLACE_TEXT,
                    {"path": "src/bug.py", "old": "return False", "new": "return True"},
                )
            )
            self.assertFalse(replaced.terminated)
            tested = environment.step(AgentAction(ActionKind.RUN_TESTS))
            self.assertTrue(tested.terminated)
            self.assertTrue(tested.test_result.passed)
            self.assertEqual(environment.changed_files(), ("src/bug.py",))
        finally:
            environment.close()

        start_argv = runner.calls[0][0]
        self.assertIn("--network", start_argv)
        self.assertIn("none", start_argv)
        self.assertIn("--cap-drop", start_argv)
        self.assertTrue(any(call[1] == self.spec.test_patch for call in runner.calls))
        self.assertEqual(runner.calls[-1][0][:3], ("docker", "rm", "--force"))

    def test_verifier_owned_test_patch_files_cannot_be_modified(self) -> None:
        runner = FakeDockerRunner(self.base_commit)
        environment = DockerSandboxEnvironment(self.spec, DockerSandboxConfig(), runner)
        try:
            environment.reset(self.task)
            result = environment.step(
                AgentAction(
                    ActionKind.REPLACE_TEXT,
                    {"path": "tests/test_bug.py", "old": "x", "new": "y"},
                )
            )
            self.assertTrue(result.terminated)
            self.assertEqual(
                result.violation,
                "invalid_action:replace_text:protected_test_file",
            )
            self.assertEqual(environment.changed_files(), ())
        finally:
            environment.close()

    def test_finish_without_an_edit_is_refused_before_any_verifier_rerun(self) -> None:
        runner = FakeDockerRunner(self.base_commit)
        environment = DockerSandboxEnvironment(self.spec, DockerSandboxConfig(), runner)
        try:
            environment.reset(self.task)
            refused = environment.step(AgentAction(ActionKind.FINISH))
            runs_after_refusal = runner.test_runs
            environment.step(
                AgentAction(
                    ActionKind.REPLACE_TEXT,
                    {"path": "src/bug.py", "old": "return False", "new": "return True"},
                )
            )
            finished = environment.step(AgentAction(ActionKind.FINISH))
        finally:
            environment.close()

        self.assertFalse(refused.terminated)
        self.assertTrue(refused.observation.startswith("Tool error: finish refused"))
        self.assertIn("Tests failed", refused.observation)
        # The unrepaired baseline failure is still valid evidence, so it is quoted
        # instead of paying for a second verifier run inside the container.
        self.assertEqual(runs_after_refusal, 1)
        self.assertTrue(finished.terminated)

    def test_finish_after_a_patch_that_did_not_fix_the_failure_is_refused(self) -> None:
        runner = _AlwaysFailingDockerRunner(self.base_commit)
        environment = DockerSandboxEnvironment(self.spec, DockerSandboxConfig(), runner)
        try:
            environment.reset(self.task)
            environment.step(
                AgentAction(
                    ActionKind.REPLACE_TEXT,
                    {"path": "src/bug.py", "old": "return False", "new": "return None"},
                )
            )
            refused = environment.step(AgentAction(ActionKind.FINISH))
        finally:
            environment.close()

        self.assertFalse(refused.terminated)
        self.assertTrue(refused.observation.startswith("Tool error: finish refused"))
        self.assertIn("has been edited and the verifier still fails", refused.observation)
        self.assertIn("Tests failed", refused.observation)

    def test_a_repeated_finish_without_an_edit_does_not_rerun_the_verifier(self) -> None:
        runner = _AlwaysFailingDockerRunner(self.base_commit)
        environment = DockerSandboxEnvironment(self.spec, DockerSandboxConfig(), runner)
        task = replace(self.task, max_steps=8)
        try:
            environment.reset(task)
            environment.step(
                AgentAction(
                    ActionKind.REPLACE_TEXT,
                    {"path": "src/bug.py", "old": "return False", "new": "return None"},
                )
            )
            after_edit = environment.step(AgentAction(ActionKind.FINISH))
            runs_after_edit = runner.test_runs
            environment.step(AgentAction(ActionKind.READ_FILE, {"path": "src/bug.py"}))
            probe = environment.step(AgentAction(ActionKind.FINISH))
        finally:
            environment.close()

        # The edit forced a run; the probe that followed no edit is answered from the failure
        # that run already produced.
        self.assertIn("has been edited and the verifier still fails", after_edit.observation)
        self.assertIn("has been edited and the verifier still fails", probe.observation)
        self.assertEqual(runs_after_edit, 2)
        self.assertEqual(runner.test_runs, 2)

    def test_replace_lines_requires_read_and_updates_source(self) -> None:
        runner = FakeDockerRunner(self.base_commit)
        environment = DockerSandboxEnvironment(self.spec, DockerSandboxConfig(), runner)
        try:
            environment.reset(self.task)
            action = AgentAction(
                ActionKind.REPLACE_LINES,
                {"path": "src/bug.py", "start_line": 7, "end_line": 7, "new": "return True"},
            )
            unread = environment.step(action)
            environment.step(AgentAction(ActionKind.READ_FILE, {"path": "src/bug.py"}))
            replaced = environment.step(action)

            self.assertIn("requires reading the target file first", unread.observation)
            self.assertEqual(replaced.observation, "Updated src/bug.py.")
            self.assertEqual(environment.changed_files(), ("src/bug.py",))
        finally:
            environment.close()

    def test_replace_lines_cannot_modify_verifier_owned_files(self) -> None:
        runner = FakeDockerRunner(self.base_commit)
        environment = DockerSandboxEnvironment(self.spec, DockerSandboxConfig(), runner)
        try:
            environment.reset(self.task)
            environment.step(AgentAction(ActionKind.READ_FILE, {"path": "tests/test_bug.py"}))
            result = environment.step(
                AgentAction(
                    ActionKind.REPLACE_LINES,
                    {"path": "tests/test_bug.py", "start_line": 1, "end_line": 1, "new": "pass"},
                )
            )

            self.assertTrue(result.terminated)
            self.assertEqual(
                result.violation,
                "invalid_action:replace_lines:protected_test_file",
            )
        finally:
            environment.close()

    def test_timed_out_test_action_terminates_the_episode(self) -> None:
        runner = FakeDockerRunner(self.base_commit)
        runner.timed_out_test_runs.add(2)
        environment = DockerSandboxEnvironment(self.spec, DockerSandboxConfig(), runner)
        try:
            environment.reset(self.task)
            result = environment.step(AgentAction(ActionKind.RUN_TESTS))

            self.assertTrue(result.terminated)
            self.assertTrue(result.test_result.timed_out)
            self.assertFalse(result.test_result.passed)
        finally:
            environment.close()

    def test_path_escape_is_rejected_before_docker_exec(self) -> None:
        runner = FakeDockerRunner(self.base_commit)
        environment = DockerSandboxEnvironment(self.spec, DockerSandboxConfig(), runner)
        try:
            environment.reset(self.task)
            call_count = len(runner.calls)
            result = environment.step(AgentAction(ActionKind.READ_FILE, {"path": "../secret"}))
            self.assertTrue(result.terminated)
            self.assertEqual(len(runner.calls), call_count)
        finally:
            environment.close()

    def test_missing_file_is_a_recoverable_tool_error(self) -> None:
        runner = FakeDockerRunner(self.base_commit)
        environment = DockerSandboxEnvironment(self.spec, DockerSandboxConfig(), runner)
        try:
            environment.reset(self.task)
            result = environment.step(AgentAction(ActionKind.READ_FILE, {"path": "missing.py"}))
            self.assertFalse(result.terminated)
            self.assertIsNone(result.violation)
            self.assertIn("Tool error: read_file failed", result.observation)
            self.assertEqual(environment.violations, [])
        finally:
            environment.close()

    def test_missing_tool_argument_is_recoverable(self) -> None:
        runner = FakeDockerRunner(self.base_commit)
        environment = DockerSandboxEnvironment(self.spec, DockerSandboxConfig(), runner)
        try:
            environment.reset(self.task)
            result = environment.step(AgentAction(ActionKind.SEARCH_TEXT))

            self.assertFalse(result.terminated)
            self.assertIsNone(result.violation)
            self.assertIn("Tool error: query must be a non-empty string", result.observation)
            self.assertEqual(environment.violations, [])
        finally:
            environment.close()

    def test_search_text_returns_repository_matches(self) -> None:
        runner = FakeDockerRunner(self.base_commit)
        environment = DockerSandboxEnvironment(self.spec, DockerSandboxConfig(), runner)
        try:
            environment.reset(self.task)
            result = environment.step(AgentAction(ActionKind.SEARCH_TEXT, {"query": "needle"}))
            self.assertFalse(result.terminated)
            self.assertIsNone(result.violation)
            self.assertEqual(result.observation, "src/bug.py:7:needle\n")
        finally:
            environment.close()

    def test_repeated_search_is_recoverable_and_skips_second_exec(self) -> None:
        runner = FakeDockerRunner(self.base_commit)
        environment = DockerSandboxEnvironment(self.spec, DockerSandboxConfig(), runner)
        try:
            environment.reset(self.task)
            action = AgentAction(ActionKind.SEARCH_TEXT, {"query": "needle"})
            environment.step(action)
            call_count = len(runner.calls)
            result = environment.step(action)

            self.assertEqual(len(runner.calls), call_count)
            self.assertFalse(result.terminated)
            self.assertIsNone(result.violation)
            self.assertIn("Tool error: do not repeat a search_text query", result.observation)
            self.assertIn("You have 3 tool steps left", result.observation)
            self.assertIn("prioritize an evidence-backed source edit", result.observation)
        finally:
            environment.close()

    def test_search_text_script_executes_as_valid_python(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "example.py").write_text("before\nneedle\nafter\n", encoding="utf-8")
            completed = subprocess.run(
                (sys.executable, "-c", DockerSandboxEnvironment._SEARCH_TEXT_SCRIPT, "needle"),
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "example.py:2:needle\n")

    def test_search_text_marks_path_matches_and_ignores_egg_info(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "moto/stepfunctions/models.py")
            source.parent.mkdir(parents=True)
            source.write_text("source content\n", encoding="utf-8")
            metadata = Path(directory, "moto.egg-info/SOURCES.txt")
            metadata.parent.mkdir(parents=True)
            metadata.write_text("moto/stepfunctions/models.py\n", encoding="utf-8")
            completed = subprocess.run(
                (
                    sys.executable,
                    "-c",
                    DockerSandboxEnvironment._SEARCH_TEXT_SCRIPT,
                    "moto/stepfunctions/models.py",
                ),
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "PATH_MATCH:moto/stepfunctions/models.py\n")

    def test_search_text_re_queries_a_path_only_the_test_spells_out(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            urls = Path(directory, "moto/moto_api/_internal/urls.py")
            urls.parent.mkdir(parents=True)
            urls.write_text(
                'url_paths = {\n    "{0}/moto-api/$": dashboard,\n}\n',
                encoding="utf-8",
            )
            test_file = Path(directory, "tests/test_config.py")
            test_file.parent.mkdir(parents=True)
            test_file.write_text(
                "from moto.moto_api._internal.urls import url_paths\n\n\ndef test_api():\n"
                '    resp = get("/moto-api/config")\n',
                encoding="utf-8",
            )
            completed = subprocess.run(
                (
                    sys.executable,
                    "-c",
                    DockerSandboxEnvironment._SEARCH_TEXT_SCRIPT,
                    "/moto-api/config",
                ),
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                'tests/test_config.py:5:    resp = get("/moto-api/config")',
                'No implementation file contains "/moto-api/config". Shorter query "moto-api" '
                "matches implementation files:",
                'moto/moto_api/_internal/urls.py:2:    "{0}/moto-api/$": dashboard,',
                "IMPLEMENTATION_CANDIDATE:moto/moto_api/_internal/urls.py",
            ],
        )

    def test_search_text_suggests_close_path_for_obsolete_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "moto/dynamodb/models/__init__.py")
            source.parent.mkdir(parents=True)
            source.write_text("source content\n", encoding="utf-8")
            completed = subprocess.run(
                (
                    sys.executable,
                    "-c",
                    DockerSandboxEnvironment._SEARCH_TEXT_SCRIPT,
                    "moto/dynamodb/models.py",
                ),
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("No exact matches for: moto/dynamodb/models.py", completed.stdout)
        self.assertIn("SUGGESTED_PATH:moto/dynamodb/models/__init__.py", completed.stdout)

    def test_read_file_script_numbers_a_selected_line_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "example.py").write_text(
                "one\ntwo\nthree\nfour\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                (
                    sys.executable,
                    "-c",
                    DockerSandboxEnvironment._READ_FILE_SCRIPT,
                    "example.py",
                    "2",
                    "3",
                ),
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            completed.stdout,
            "2: two\n"
            "3: three\n"
            "[read_file lines 2-3: file has 4 lines; "
            "continue with read_file start_line=4 end_line=4]\n",
        )

    def test_read_file_script_marks_an_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "example.py").write_text("", encoding="utf-8")
            completed = subprocess.run(
                (
                    sys.executable,
                    "-c",
                    DockerSandboxEnvironment._READ_FILE_SCRIPT,
                    "example.py",
                ),
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "[file is empty]\n")

    def test_read_file_script_reports_when_the_character_budget_truncates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "example.py").write_text(
                "\n".join(f"line-{number}-{'x' * 60}" for number in range(1, 201)) + "\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                (
                    sys.executable,
                    "-c",
                    DockerSandboxEnvironment._READ_FILE_SCRIPT,
                    "example.py",
                ),
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        lines = completed.stdout.splitlines()
        self.assertEqual(lines[0], "1: line-1-" + "x" * 60)
        self.assertIn("[read_file lines 1-", lines[-1])
        self.assertIn("file has 200 lines", lines[-1])
        self.assertLess(len(completed.stdout), 8_400)

    def test_search_text_script_caps_matches_per_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "src").mkdir(parents=True)
            Path(directory, "src", "noisy.py").write_text(
                "needle\n" * 20,
                encoding="utf-8",
            )
            Path(directory, "src", "other.py").write_text("needle\n", encoding="utf-8")
            completed = subprocess.run(
                (sys.executable, "-c", DockerSandboxEnvironment._SEARCH_TEXT_SCRIPT, "needle"),
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        matches = completed.stdout.splitlines()
        self.assertEqual(len(matches), 6)
        self.assertEqual(sum(line.startswith("src/noisy.py:") for line in matches), 5)
        self.assertEqual(sum(line.startswith("src/other.py:") for line in matches), 1)

    def test_replace_lines_script_preserves_the_selected_block_newline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "example.py")
            path.write_text("one\ntwo\nthree\n", encoding="utf-8")
            completed = subprocess.run(
                (
                    sys.executable,
                    "-c",
                    DockerSandboxEnvironment._REPLACE_LINES_SCRIPT,
                    "example.py",
                    "2",
                    "2",
                    "replacement",
                ),
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
            updated = path.read_text(encoding="utf-8")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(updated, "one\nreplacement\nthree\n")

    def test_replace_text_script_matches_the_local_mismatch_message(self) -> None:
        mismatches = (
            ("def is_fixed():\n    return False\n", "def is_fixed(): return False"),
            ("def is_fixed():\n    return False\n", "absent_call()"),
            ("value = 1\nother = 2\nvalue = 1\n", "value = 1"),
        )
        for content, old in mismatches:
            with self.subTest(old=old), tempfile.TemporaryDirectory() as directory:
                path = Path(directory, "example.py")
                path.write_text(content, encoding="utf-8")
                completed = subprocess.run(
                    (
                        sys.executable,
                        "-c",
                        DockerSandboxEnvironment._REPLACE_TEXT_SCRIPT,
                        "example.py",
                        old,
                        "replacement",
                    ),
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                self.assertEqual(completed.returncode, 1)
                self.assertEqual(path.read_text(encoding="utf-8"), content)
                self.assertEqual(
                    completed.stderr.strip(),
                    replace_text_mismatch_message(content, old),
                )

    def test_search_text_script_matches_the_local_candidate_hint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory, "repo")
            (repository / "tests").mkdir(parents=True)
            (repository / "pkg" / "core").mkdir(parents=True)
            (repository / "pkg" / "__init__.py").write_text("", encoding="utf-8")
            (repository / "pkg" / "core" / "__init__.py").write_text("", encoding="utf-8")
            (repository / "pkg" / "core" / "config.py").write_text("BATCH = 1\n", encoding="utf-8")
            (repository / "tests" / "test_config.py").write_text(
                'from pkg.core.config import BATCH\n\n\ndef test_api():\n'
                '    response = get("/moto-api/config")\n',
                encoding="utf-8",
            )
            completed = subprocess.run(
                (
                    sys.executable,
                    "-c",
                    DockerSandboxEnvironment._SEARCH_TEXT_SCRIPT,
                    "/moto-api/config",
                ),
                cwd=repository,
                capture_output=True,
                text=True,
                check=False,
            )
            expected = search_repository(repository, "/moto-api/config")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), expected)
        self.assertIn("IMPLEMENTATION_CANDIDATE:pkg/core/config.py", expected)

    def test_edit_scripts_refuse_a_python_edit_that_would_not_parse(self) -> None:
        content = "def is_fixed():\n    value = 1\n    return False\n"
        broken = "def is_fixed():\n    value = 1\n        return True\n"
        cases = (
            (
                DockerSandboxEnvironment._REPLACE_LINES_SCRIPT,
                ("example.py", "3", "3", "        return True"),
            ),
            (
                DockerSandboxEnvironment._REPLACE_TEXT_SCRIPT,
                ("example.py", "    return False", "        return True"),
            ),
        )
        for script, arguments in cases:
            with self.subTest(script=arguments), tempfile.TemporaryDirectory() as directory:
                path = Path(directory, "example.py")
                path.write_text(content, encoding="utf-8")
                completed = subprocess.run(
                    (sys.executable, "-c", script, *arguments),
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                expected = python_edit_syntax_error("example.py", broken)

                self.assertEqual(completed.returncode, 1)
                self.assertEqual(path.read_text(encoding="utf-8"), content)
                self.assertEqual(completed.stderr.strip(), expected)

    def test_edit_scripts_name_the_enclosing_lines_of_a_block_left_open(self) -> None:
        content = 'url_paths = {\n    "a": 1,\n    "b": 2,\n}\n'
        broken = 'url_paths = {\n    "d": 4,\n}\n    "b": 2,\n}\n'
        cases = (
            (
                DockerSandboxEnvironment._REPLACE_LINES_SCRIPT,
                ("example.py", "1", "2", 'url_paths = {\n    "d": 4,\n}'),
            ),
            (
                DockerSandboxEnvironment._REPLACE_TEXT_SCRIPT,
                (
                    "example.py",
                    'url_paths = {\n    "a": 1,',
                    'url_paths = {\n    "d": 4,\n}',
                ),
            ),
        )
        for script, arguments in cases:
            with self.subTest(script=arguments), tempfile.TemporaryDirectory() as directory:
                path = Path(directory, "example.py")
                path.write_text(content, encoding="utf-8")
                completed = subprocess.run(
                    (sys.executable, "-c", script, *arguments),
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                expected = python_edit_syntax_error(
                    "example.py", broken, original=content, replaced_lines=(1, 2)
                )

                self.assertEqual(completed.returncode, 1)
                self.assertEqual(path.read_text(encoding="utf-8"), content)
                self.assertIn("spans lines 1-4", expected or "")
                self.assertEqual(completed.stderr.strip(), expected)

    def test_search_text_ranks_source_before_tests_and_docs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for relative in ("docs/guide.py", "tests/test_feature.py", "src/feature.py"):
                path = Path(directory, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("needle\n", encoding="utf-8")
            completed = subprocess.run(
                (sys.executable, "-c", DockerSandboxEnvironment._SEARCH_TEXT_SCRIPT, "needle"),
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                "src/feature.py:1:needle",
                "tests/test_feature.py:1:needle",
                "docs/guide.py:1:needle",
            ],
        )


if __name__ == "__main__":
    unittest.main()
