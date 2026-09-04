from __future__ import annotations

import difflib
import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .contracts import ActionKind, AgentAction, CodingTask, StepResult, TestResult
from .verifier import LocalPythonVerifier


class EnvironmentError(RuntimeError):
    pass


class ToolError(RuntimeError):
    """A recoverable tool failure that should be shown to the policy."""


def read_line_range(arguments: dict[str, Any]) -> tuple[int, int] | None:
    has_start = "start_line" in arguments
    has_end = "end_line" in arguments
    if not has_start and not has_end:
        return None
    if not has_start or not has_end:
        raise ToolError("read_file line ranges require both start_line and end_line")
    start_line = arguments["start_line"]
    end_line = arguments["end_line"]
    if (
        isinstance(start_line, bool)
        or isinstance(end_line, bool)
        or not isinstance(start_line, int)
        or not isinstance(end_line, int)
    ):
        raise ToolError("read_file line ranges must be integers")
    if start_line < 1 or end_line < start_line:
        raise ToolError("read_file requires 1 <= start_line <= end_line")
    line_count = end_line - start_line + 1
    if line_count < 20:
        raise ToolError("read_file line range must include at least 20 lines of context")
    if line_count > 400:
        raise ToolError("read_file line range cannot exceed 400 lines")
    return start_line, end_line


def select_line_range(content: str, line_range: tuple[int, int] | None) -> str:
    if line_range is None:
        return content
    start_line, end_line = line_range
    return "".join(content.splitlines(keepends=True)[start_line - 1 : end_line])


class ActionLoopGuard:
    """Reject unproductive repeated tool calls as recoverable observations."""

    def __init__(self) -> None:
        self._records: list[tuple[AgentAction, str]] = []

    def reset(self) -> None:
        self._records = []

    def rejection_for(self, action: AgentAction) -> str | None:
        if self._records:
            previous_action, previous_observation = self._records[-1]
            if previous_action == action and previous_observation.startswith("Tool error:"):
                return "do not repeat the same action after a tool error; choose another path or tool"
        if action.kind is ActionKind.SEARCH_TEXT and any(
            previous == action and not observation.startswith("Tool error:")
            for previous, observation in self._records
        ):
            return "do not repeat a search_text query that already returned a result"
        if action.kind is ActionKind.LIST_FILES and any(
            previous.kind is ActionKind.LIST_FILES for previous, _ in self._records
        ):
            return "do not repeat list_files"
        if action.kind is ActionKind.READ_FILE:
            previous_reads = [
                index
                for index, (previous, _) in enumerate(self._records)
                if previous == action
            ]
            if previous_reads:
                path = action.arguments.get("path")
                changed_after_read = any(
                    previous.kind is ActionKind.REPLACE_TEXT
                    and previous.arguments.get("path") == path
                    and observation.startswith("Updated ")
                    for previous, observation in self._records[previous_reads[-1] + 1 :]
                )
                if not changed_after_read:
                    return "do not reread an unchanged file; use search_text or inspect another file"
        return None

    def record(self, action: AgentAction, observation: str) -> None:
        self._records.append((action, observation))


@runtime_checkable
class CodingEnvironment(Protocol):
    """Runtime contract used by rollout collection, independent of sandbox type."""

    baseline_result: TestResult | None
    tool_calls: int
    violations: list[str]

    def reset(self, task: CodingTask) -> str: ...

    def step(self, action: AgentAction) -> StepResult: ...

    def finalize(self) -> TestResult: ...

    def changed_files(self) -> tuple[str, ...]: ...

    def patch_is_valid(self) -> bool: ...

    def close(self) -> None: ...


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
        self._action_loop_guard = ActionLoopGuard()

    def reset(self, task: CodingTask) -> str:
        self.close()
        if task.fixture_path is None:
            raise EnvironmentError(f"task {task.task_id} does not declare a local fixture path")
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
        self._action_loop_guard.reset()
        self._initial_hashes = self._file_hashes()
        self.baseline_result = self.verifier.run(self.repository, task.test_command)
        self.last_test_result = self.baseline_result
        if self.baseline_result.passed:
            self.close()
            raise EnvironmentError(f"task {task.task_id} is invalid: baseline tests already pass")
        return f"Baseline verifier result:\n{self._test_observation(self.baseline_result)}"

    def step(self, action: AgentAction) -> StepResult:
        task, repository = self._require_active()
        if self.steps >= task.max_steps:
            return StepResult("Step budget exhausted.", True, self.last_test_result)
        self.steps += 1
        self.tool_calls += 1
        rejection = self._action_loop_guard.rejection_for(action)
        if rejection:
            result = StepResult(f"Tool error: {rejection}", False, self.last_test_result)
            self._action_loop_guard.record(action, result.observation)
            return result
        try:
            if action.kind is ActionKind.LIST_FILES:
                files = sorted(
                    str(path.relative_to(repository))
                    for path in repository.rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts
                )
                result = StepResult(self._bounded_listing("\n".join(files)), False)
            elif action.kind is ActionKind.SEARCH_TEXT:
                query = self._required_string(action.arguments, "query")
                if len(query) > 200:
                    raise ToolError("search_text query must be at most 200 characters")
                matches: list[str] = []
                repository_paths: list[str] = []
                query_folded = query.casefold()
                for path in repository.rglob("*"):
                    if not path.is_file() or any(
                        part in {".git", "__pycache__", ".pytest_cache"}
                        or part.endswith(".egg-info")
                        for part in path.parts
                    ):
                        continue
                    relative = path.relative_to(repository).as_posix()
                    repository_paths.append(relative)
                    if query_folded in relative.casefold():
                        matches.append(f"PATH_MATCH:{relative}")
                    if path.stat().st_size > 1_000_000:
                        continue
                    try:
                        lines = path.read_text(encoding="utf-8").splitlines()
                    except (OSError, UnicodeError):
                        continue
                    for line_number, line in enumerate(lines, start=1):
                        if query_folded in line.casefold():
                            matches.append(f"{relative}:{line_number}:{line[:300]}")
                            if len(matches) >= 100:
                                break
                    if len(matches) >= 100:
                        break
                if matches:
                    observation = "\n".join(matches)
                else:
                    candidates = {relative.casefold(): relative for relative in repository_paths}
                    suggestions = difflib.get_close_matches(
                        query_folded,
                        candidates,
                        n=8,
                        cutoff=0.45,
                    )
                    rendered = [f"No exact matches for: {query}"]
                    rendered.extend(f"SUGGESTED_PATH:{candidates[item]}" for item in suggestions)
                    observation = "\n".join(rendered)
                result = StepResult(observation, False)
            elif action.kind is ActionKind.READ_FILE:
                path = self._resolve_repository_path(action.arguments.get("path"))
                content = path.read_text(encoding="utf-8")
                content = select_line_range(content, read_line_range(action.arguments))
                result = StepResult(content[:20_000], False)
            elif action.kind is ActionKind.REPLACE_TEXT:
                path = self._resolve_repository_path(action.arguments.get("path"))
                old = self._required_string(action.arguments, "old")
                new = self._required_string(action.arguments, "new", allow_empty=True)
                content = path.read_text(encoding="utf-8")
                occurrences = content.count(old)
                if occurrences != 1:
                    raise ToolError(f"replace_text requires exactly one match, found {occurrences}")
                path.write_text(content.replace(old, new, 1), encoding="utf-8")
                result = StepResult(f"Updated {path.relative_to(repository)}.", False)
            elif action.kind is ActionKind.RUN_TESTS:
                result = self.verifier.run(repository, task.test_command)
                self.last_test_result = result
                result = StepResult(self._test_observation(result), result.passed, result)
            elif action.kind is ActionKind.FINISH:
                result = self.verifier.run(repository, task.test_command)
                self.last_test_result = result
                result = StepResult(self._test_observation(result), True, result)
            else:
                raise EnvironmentError(f"unsupported action: {action.kind.value}")
        except (ToolError, OSError, UnicodeError) as exc:
            result = StepResult(f"Tool error: {exc}", False, self.last_test_result)
        except EnvironmentError as exc:
            violation = f"invalid_action:{type(exc).__name__}"
            self.violations.append(violation)
            result = StepResult(str(exc), True, self.last_test_result, violation)
        self._action_loop_guard.record(action, result.observation)
        return result

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

    def patch_is_valid(self) -> bool:
        if self.repository is None:
            return False
        changed = self.changed_files()
        if not changed:
            return False
        try:
            for relative in changed:
                path = self.repository / relative
                if not path.is_file():
                    return False
                if path.suffix == ".py":
                    compile(path.read_text(encoding="utf-8"), relative, "exec")
        except (OSError, SyntaxError, UnicodeError):
            return False
        return True

    def close(self) -> None:
        if self.workspace is not None and self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.task = None
        self.workspace = None
        self.repository = None
        self.baseline_result = None
        self.last_test_result = None
        self._initial_hashes = {}
        self._action_loop_guard.reset()

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
            raise ToolError(f"file does not exist: {raw}")
        return candidate

    def _file_hashes(self) -> dict[str, str]:
        _, repository = self._require_active()
        result: dict[str, str] = {}
        for path in repository.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            result[str(path.relative_to(repository))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    @staticmethod
    def _bounded_listing(content: str, limit: int = 20_000) -> str:
        if len(content) <= limit:
            return content
        half = (limit - 100) // 2
        return (
            content[:half]
            + "\n...[file list truncated; use search_text or run_tests to locate relevant code]...\n"
            + content[-half:]
        )

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
