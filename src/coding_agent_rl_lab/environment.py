from __future__ import annotations

import difflib
import hashlib
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, Sequence, runtime_checkable

READ_FILE_DEFAULT_LINES = 200
READ_FILE_MAX_CHARS = 8_000
SEARCH_TOTAL_LIMIT = 100
SEARCH_PER_FILE_LIMIT = 5
SEARCH_FALLBACK_LIMIT = 20
_DOCUMENTATION_DIRECTORIES = frozenset({"docs", "doc", "examples", "example"})
_DOCUMENTATION_NAMES = (
    "changelog",
    "implementation_coverage",
    "contributing",
    "readme",
    "notice",
    "license",
    "authors",
    "news",
)

from .contracts import ActionKind, AgentAction, CodingTask, StepResult, TestResult
from .verifier import LocalPythonVerifier


class EnvironmentError(RuntimeError):
    pass


class ToolError(RuntimeError):
    """A recoverable tool failure that should be shown to the policy."""


def action_violation_code(action: AgentAction, error: EnvironmentError) -> str:
    """Return a stable answer-free category for a terminating action violation."""

    message = str(error)
    if message.startswith("cannot modify verifier-owned test file:"):
        category = "protected_test_file"
    elif message == "path escapes repository workspace":
        category = "path_escape"
    elif message.startswith("unsupported action:"):
        category = "unsupported_action"
    else:
        category = "environment_state"
    return f"invalid_action:{action.kind.value}:{category}"


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


def render_numbered_window(
    content: str,
    line_range: tuple[int, int] | None = None,
    *,
    max_lines: int = READ_FILE_DEFAULT_LINES,
    max_chars: int = READ_FILE_MAX_CHARS,
) -> str:
    """Render a line-numbered, self-describing window over one file.

    Line numbers are what let a policy derive `replace_text` and `replace_lines`
    arguments from an observation, and the closing footer is what stops a truncated
    read from looking like the whole file.
    """

    lines = content.splitlines()
    total = len(lines)
    if total == 0:
        return "[file is empty]"
    if line_range is None:
        first, last = 1, min(total, max_lines)
    else:
        first, last = max(1, line_range[0]), min(total, line_range[1])
    if last < first:
        return f"[no lines in range: the file has {total} lines]"
    rendered: list[str] = []
    used_chars = 0
    shown_last = first - 1
    for number in range(first, last + 1):
        line = f"{number}: {lines[number - 1]}"
        if rendered and used_chars + len(line) + 1 > max_chars:
            break
        rendered.append(line)
        used_chars += len(line) + 1
        shown_last = number
    if shown_last >= total:
        return "\n".join(rendered)
    notes = ["character budget reached"] if shown_last < last else []
    notes.append(f"file has {total} lines")
    notes.append(
        "continue with read_file start_line="
        f"{shown_last + 1} end_line={min(total, shown_last + READ_FILE_DEFAULT_LINES)}"
    )
    return "\n".join((*rendered, f"[read_file lines {first}-{shown_last}: {'; '.join(notes)}]"))


def search_location_rank(parts: Sequence[str]) -> int:
    """Rank implementation files ahead of tests and prose; lower is better."""

    if any(part in _DOCUMENTATION_DIRECTORIES for part in parts):
        return 2
    name = parts[-1] if parts else ""
    if name.endswith((".md", ".rst", ".txt")) or name.startswith(_DOCUMENTATION_NAMES):
        return 2
    if any(part == "tests" or part.startswith("test") for part in parts):
        return 1
    return 0


def repository_file_listing(repository: Path) -> tuple[str, ...]:
    """Deterministic, workspace-relative listing of searchable repository files."""

    return tuple(
        sorted(
            path.relative_to(repository).as_posix()
            for path in repository.rglob("*")
            if path.is_file()
            and not any(
                part in {".git", "__pycache__", ".pytest_cache"} or part.endswith(".egg-info")
                for part in path.relative_to(repository).parts
            )
        )
    )


def _collect_search_matches(
    repository: Path,
    query: str,
    *,
    listing: Sequence[str],
    per_file_limit: int,
) -> list[str]:
    folded = query.casefold()
    ranked: list[tuple[int, int, str, int, str]] = []
    for relative in listing:
        parts = tuple(part.casefold() for part in PurePosixPath(relative).parts)
        rank = search_location_rank(parts)
        if folded in relative.casefold():
            ranked.append((rank, 0, relative, 0, f"PATH_MATCH:{relative}"))
        path = repository / relative
        try:
            if path.stat().st_size > 1_000_000:
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        matched_in_file = 0
        for number, line in enumerate(lines, start=1):
            if folded not in line.casefold():
                continue
            ranked.append((rank, 1, relative, number, f"{relative}:{number}:{line[:300]}"))
            matched_in_file += 1
            if matched_in_file >= per_file_limit:
                break
    ranked.sort(key=lambda item: item[:4])
    return [item[4] for item in ranked]


def search_repository(
    repository: Path,
    query: str,
    *,
    total_limit: int = SEARCH_TOTAL_LIMIT,
    per_file_limit: int = SEARCH_PER_FILE_LIMIT,
    fallback_limit: int = SEARCH_FALLBACK_LIMIT,
    max_chars: int = READ_FILE_MAX_CHARS,
) -> str:
    """Return a ranked, per-file-capped literal search with one token fallback.

    An issue-derived sentence almost never matches source text, and an uncapped match
    list lets one large file consume the entire result budget, so a failed query is
    retried with its longest token instead of spending another agent step on nothing.
    The result is trimmed to `max_chars` from the top so the highest-ranked matches
    survive the prompt budget instead of the tail the model happens to receive.
    """

    listing = repository_file_listing(repository)
    matches = _collect_search_matches(
        repository,
        query,
        listing=listing,
        per_file_limit=per_file_limit,
    )
    if matches:
        return "\n".join(
            _bounded_search_lines(matches[:total_limit], max_chars=max_chars)
        )
    rendered = [f"No exact matches for: {query}"]
    tokens = sorted(
        {token for token in query.split() if len(token) >= 4},
        key=len,
        reverse=True,
    )
    for token in tokens[:1]:
        token_matches = _collect_search_matches(
            repository,
            token,
            listing=listing,
            per_file_limit=per_file_limit,
        )
        if token_matches:
            rendered.append(f"Longest token in the query: {token}")
            rendered.extend(
                _bounded_search_lines(
                    token_matches[:fallback_limit],
                    max_chars=max_chars,
                )
            )
            break
    else:
        candidates = {path.casefold(): path for path in listing}
        rendered.extend(
            f"SUGGESTED_PATH:{candidates[item]}"
            for item in difflib.get_close_matches(query.casefold(), candidates, n=8, cutoff=0.45)
        )
    return "\n".join(_bounded_search_lines(rendered, max_chars=max_chars))


def _bounded_search_lines(lines: Sequence[str], *, max_chars: int) -> list[str]:
    """Keep the best-ranked whole match lines, and say so when the budget cut them."""

    rendered: list[str] = []
    used_chars = 0
    for line in lines:
        if rendered and used_chars + len(line) + 1 > max_chars:
            break
        rendered.append(line)
        used_chars += len(line) + 1
    if len(rendered) < len(lines):
        rendered.append(
            f"[search_text: showing {len(rendered)} of {len(lines)} matches; "
            "narrow the query or read the first file]"
        )
    return rendered


def replace_line_range(
    content: str,
    *,
    start_line: int,
    end_line: int,
    new: str,
) -> str:
    """Replace a small inclusive one-based line range while preserving its final newline."""

    if isinstance(start_line, bool) or isinstance(end_line, bool):
        raise ToolError("replace_lines line ranges must be integers")
    if not isinstance(start_line, int) or not isinstance(end_line, int):
        raise ToolError("replace_lines line ranges must be integers")
    if start_line < 1 or end_line < start_line:
        raise ToolError("replace_lines requires 1 <= start_line <= end_line")
    if end_line - start_line + 1 > 80:
        raise ToolError("replace_lines cannot replace more than 80 lines")
    if not isinstance(new, str):
        raise ToolError("new must be a string")

    lines = content.splitlines(keepends=True)
    if end_line > len(lines):
        raise ToolError(
            f"replace_lines end_line {end_line} exceeds file length {len(lines)}"
        )
    selected = "".join(lines[start_line - 1 : end_line])
    replacement = new
    if replacement and selected.endswith("\n") and not replacement.endswith(("\n", "\r")):
        replacement += "\n"
    return "".join(lines[: start_line - 1]) + replacement + "".join(lines[end_line:])


class ActionLoopGuard:
    """Reject unproductive repeated tool calls as recoverable observations.

    A weak policy often answers a rejection by repeating itself and then burns the whole
    step budget on refusals. Rejections therefore escalate: the first one is the plain
    message, and later ones add the state the policy is failing to track.
    """

    def __init__(self) -> None:
        self._records: list[tuple[AgentAction, str]] = []
        self.rejection_count = 0
        self.consecutive_rejections = 0

    def reset(self) -> None:
        self._records = []
        self.rejection_count = 0
        self.consecutive_rejections = 0

    @property
    def patched(self) -> bool:
        return any(observation.startswith("Updated ") for _, observation in self._records)

    def rejection_for(self, action: AgentAction) -> str | None:
        base = self._base_rejection(action)
        if base is None:
            return None
        self.rejection_count += 1
        self.consecutive_rejections += 1
        if self.consecutive_rejections == 1:
            return base
        return f"{base}. {self._recovery_directive()}"

    def record(self, action: AgentAction, observation: str) -> None:
        if not observation.startswith("Tool error:"):
            self.consecutive_rejections = 0
        self._records.append((action, observation))

    def _base_rejection(self, action: AgentAction) -> str | None:
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
                    previous.kind in {ActionKind.REPLACE_TEXT, ActionKind.REPLACE_LINES}
                    and previous.arguments.get("path") == path
                    and observation.startswith("Updated ")
                    for previous, observation in self._records[previous_reads[-1] + 1 :]
                )
                if not changed_after_read:
                    return "do not reread an unchanged file; use search_text or inspect another file"
        return None

    def _recovery_directive(self) -> str:
        parts = [f"Repeated rejection {self.consecutive_rejections} times in a row."]
        if not self.patched:
            parts.append("No patch has been applied yet.")
        read_files: list[str] = []
        queries: list[str] = []
        for previous, _ in self._records:
            if previous.kind is ActionKind.READ_FILE:
                path = previous.arguments.get("path")
                if isinstance(path, str) and path not in read_files:
                    read_files.append(path)
            elif previous.kind is ActionKind.SEARCH_TEXT:
                query = previous.arguments.get("query")
                if isinstance(query, str) and query not in queries:
                    queries.append(query)
        if read_files:
            parts.append("Files already read: " + ", ".join(read_files[:5]) + ".")
        if queries:
            parts.append("Queries already used: " + "; ".join(queries[:5]) + ".")
        parts.append(
            "Do not issue it again. Either edit a file you already read with replace_lines "
            "on a small range, or call finish."
        )
        return " ".join(parts)


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

    def graded_targets(self) -> tuple[tuple[str, ...], tuple[str, ...]]: ...

    @property
    def loop_rejections(self) -> int: ...

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
        self._read_files: set[str] = set()
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
        self._read_files = set()
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
                result = StepResult(search_repository(repository, query), False)
            elif action.kind is ActionKind.READ_FILE:
                path = self._resolve_repository_path(action.arguments.get("path"))
                content = path.read_text(encoding="utf-8")
                self._read_files.add(path.relative_to(repository).as_posix())
                result = StepResult(
                    render_numbered_window(content, read_line_range(action.arguments)),
                    False,
                )
            elif action.kind is ActionKind.REPLACE_TEXT:
                path = self._resolve_repository_path(action.arguments.get("path"))
                old = self._required_string(action.arguments, "old")
                new = self._required_string(action.arguments, "new", allow_empty=True)
                content = path.read_text(encoding="utf-8")
                occurrences = content.count(old)
                if occurrences != 1:
                    raise ToolError(f"replace_text requires exactly one match, found {occurrences}")
                path.write_text(content.replace(old, new, 1), encoding="utf-8")
                self._read_files.discard(path.relative_to(repository).as_posix())
                result = StepResult(f"Updated {path.relative_to(repository)}.", False)
            elif action.kind is ActionKind.REPLACE_LINES:
                path = self._resolve_repository_path(action.arguments.get("path"))
                relative = path.relative_to(repository).as_posix()
                if relative not in self._read_files:
                    raise ToolError("replace_lines requires reading the target file first")
                new = self._required_string(action.arguments, "new", allow_empty=True)
                content = path.read_text(encoding="utf-8")
                updated = replace_line_range(
                    content,
                    start_line=action.arguments.get("start_line"),
                    end_line=action.arguments.get("end_line"),
                    new=new,
                )
                path.write_text(updated, encoding="utf-8")
                self._read_files.discard(relative)
                result = StepResult(f"Updated {relative}.", False)
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
            violation = action_violation_code(action, exc)
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

    def graded_targets(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Fixtures grade whatever their own trusted test command covers."""

        return (), ()

    @property
    def loop_rejections(self) -> int:
        """Count of policy actions the loop guard refused as unproductive repeats."""

        return self._action_loop_guard.rejection_count

    def close(self) -> None:
        if self.workspace is not None and self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.task = None
        self.workspace = None
        self.repository = None
        self.baseline_result = None
        self.last_test_result = None
        self._initial_hashes = {}
        self._read_files = set()
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
            raise ToolError("path must be a non-empty string")
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
            raise ToolError(f"{name} must be a {'string' if allow_empty else 'non-empty string'}")
        return value

    @staticmethod
    def _test_observation(result: TestResult) -> str:
        status = "passed" if result.passed else "failed"
        detail = result.stderr or result.stdout
        return f"Tests {status} (exit={result.exit_code}).\n{detail[-4000:]}"
