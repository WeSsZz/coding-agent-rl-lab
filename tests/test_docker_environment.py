from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from coding_agent_rl_lab.contracts import ActionKind, AgentAction, CodingTask, DatasetSplit
from coding_agent_rl_lab.docker_environment import (
    CommandExecution,
    DockerSandboxConfig,
    DockerSandboxEnvironment,
    DockerTaskSpec,
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
            self.assertEqual(result.violation, "invalid_action:EnvironmentError")
            self.assertEqual(environment.changed_files(), ())
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

    def test_read_file_script_can_select_a_line_range(self) -> None:
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
        self.assertEqual(completed.stdout, "two\nthree\n")

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
