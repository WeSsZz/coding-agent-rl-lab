from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .contracts import ActionKind, AgentAction, CodingTask, ExecutionManifest, StepResult, TestResult
from .verifier import LocalPythonVerifier


class EnvironmentError(RuntimeError):
    pass


class LocalFixtureEnvironment:
    """A deliberately narrow environment for infrastructure tests, not untrusted code."""

    def __init__(self, project_root: str | Path, verifier: LocalPythonVerifier | None = None) -> None:
        self.project_root = Path(project_root).resolve()
        self.verifier = verifier or LocalPythonVerifier()
        self.task: CodingTask | None = None
        self.workspace: Path | None = None
        self.repository: Path | None = None
        self.baseline_result: TestResult | None = None
        self.last_test_result: TestResult | None = None
        self.steps = 0
        self.tool_calls = 0
        self.violations: list[str] = []
        self._initial_hashes: dict[str, str] = {}

    @property
    def execution_manifest(self) -> ExecutionManifest:
        return ExecutionManifest(
            environment_id="local-fixture-environment",
            environment_version="1",
            verifier_id="local-python-verifier",
            verifier_version="1",
            sandbox_provider="trusted-fixture-tempdir",
            sandbox_version="1",
            tool_contract="coding-action-v1",
        )

    def reset(self, task: CodingTask) -> str:
        self.close()
        source = self._resolve_project_path(task.fixture_path) / "repo"
        if not source.is_dir():
            raise EnvironmentError(f"fixture repository does not exist: {task.fixture_path}")
        self.workspace = Path(tempfile.mkdtemp(prefix=f"coding-agent-{task.task_id}-")).resolve()
        self.repository = self.workspace / "repo"
        shutil.copytree(source, self.repository)
        self.repository = self.repository.resolve()
        self.task = task
        self.steps = 0
        self.tool_calls = 0
        self.violations = []
        self._initial_hashes = self._file_hashes()
        self.baseline_result = self.verifier.run(self.repository, task.test_command)
        self.last_test_result = self.baseline_result
        if self.baseline_result.passed:
            self.close()
            raise EnvironmentError(f"task {task.task_id} is invalid: baseline tests already pass")
        return f"Issue: {task.issue}\nBaseline tests fail as expected."

    def step(self, action: AgentAction) -> StepResult:
        task, repository = self._require_active()
        if self.steps >= task.max_steps:
            return StepResult("Step budget exhausted.", True, self.last_test_result)
        self.steps += 1
        self.tool_calls += 1
        try:
            if action.kind is ActionKind.LIST_FILES:
                files = sorted(
                    str(path.relative_to(repository))
                    for path in repository.rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts
                )
                return StepResult("\n".join(files), False)
            if action.kind is ActionKind.READ_FILE:
                path = self._resolve_repository_path(action.arguments.get("path"))
                return StepResult(path.read_text(encoding="utf-8")[:20_000], False)
            if action.kind is ActionKind.REPLACE_TEXT:
                path = self._resolve_repository_path(action.arguments.get("path"))
                old = self._required_string(action.arguments, "old")
                new = self._required_string(action.arguments, "new", allow_empty=True)
                content = path.read_text(encoding="utf-8")
                occurrences = content.count(old)
                if occurrences != 1:
                    raise EnvironmentError(f"replace_text requires exactly one match, found {occurrences}")
                path.write_text(content.replace(old, new, 1), encoding="utf-8")
                return StepResult(f"Updated {path.relative_to(repository)}.", False)
            if action.kind is ActionKind.RUN_TESTS:
                result = self.verifier.run(repository, task.test_command)
                self.last_test_result = result
                return StepResult(self._test_observation(result), result.passed, result)
            if action.kind is ActionKind.FINISH:
                result = self.verifier.run(repository, task.test_command)
                self.last_test_result = result
                return StepResult(self._test_observation(result), True, result)
            raise EnvironmentError(f"unsupported action: {action.kind.value}")
        except (EnvironmentError, OSError, UnicodeError) as exc:
            violation = f"invalid_action:{type(exc).__name__}"
            self.violations.append(violation)
            return StepResult(str(exc), True, self.last_test_result, violation)

    def finalize(self) -> TestResult:
        task, repository = self._require_active()
        result = self.verifier.run(repository, task.test_command)
        self.last_test_result = result
        return result

    def changed_files(self) -> tuple[str, ...]:
        if self.repository is None:
            return ()
        current = self._file_hashes()
        names = set(self._initial_hashes) | set(current)
        return tuple(sorted(name for name in names if self._initial_hashes.get(name) != current.get(name)))

    def close(self) -> None:
        if self.workspace is not None and self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.task = None
        self.workspace = None
        self.repository = None
        self.baseline_result = None
        self.last_test_result = None
        self._initial_hashes = {}

    def __enter__(self) -> LocalFixtureEnvironment:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _resolve_project_path(self, relative: str) -> Path:
        candidate = (self.project_root / relative).resolve()
        if not candidate.is_relative_to(self.project_root):
            raise EnvironmentError("fixture path escapes project root")
        return candidate

    def _resolve_repository_path(self, raw: Any) -> Path:
        if not isinstance(raw, str) or not raw:
            raise EnvironmentError("path must be a non-empty string")
        _, repository = self._require_active()
        candidate = (repository / raw).resolve()
        if not candidate.is_relative_to(repository):
            raise EnvironmentError("path escapes repository workspace")
        if not candidate.is_file():
            raise EnvironmentError(f"file does not exist: {raw}")
        return candidate

    def _file_hashes(self) -> dict[str, str]:
        _, repository = self._require_active()
        result: dict[str, str] = {}
        for path in repository.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            result[str(path.relative_to(repository))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    def _require_active(self) -> tuple[CodingTask, Path]:
        if self.task is None or self.repository is None:
            raise EnvironmentError("environment is not active")
        return self.task, self.repository

    @staticmethod
    def _required_string(arguments: dict[str, Any], name: str, *, allow_empty: bool = False) -> str:
        value = arguments.get(name)
        if not isinstance(value, str) or (not allow_empty and not value):
            raise EnvironmentError(f"{name} must be a {'string' if allow_empty else 'non-empty string'}")
        return value

    @staticmethod
    def _test_observation(result: TestResult) -> str:
        status = "passed" if result.passed else "failed"
        detail = result.stderr or result.stdout
        return f"Tests {status} (exit={result.exit_code}).\n{detail[-4000:]}"
