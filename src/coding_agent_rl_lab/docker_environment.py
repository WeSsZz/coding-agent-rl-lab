from __future__ import annotations

import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Protocol

from .contracts import ActionKind, AgentAction, CodingTask, StepResult, TestResult
from .environment import (
    ActionLoopGuard,
    CodingEnvironment,
    EnvironmentError,
    ToolError,
    action_violation_code,
    failure_summary,
    is_test_path,
    premature_finish_refusal,
    python_edit_syntax_error,
    read_line_range,
    read_range_requirement_message,
    replace_line_range_bounds,
    shown_line_span,
    verifier_output_detail,
)


@dataclass(frozen=True)
class CommandExecution:
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0
    timed_out: bool = False

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class CommandRunner(Protocol):
    def run(
        self,
        argv: tuple[str, ...],
        *,
        input_text: str | None = None,
        timeout_seconds: float | None = None,
    ) -> CommandExecution: ...


class SubprocessCommandRunner:
    """Executes trusted host-side argv without a shell."""

    def run(
        self,
        argv: tuple[str, ...],
        *,
        input_text: str | None = None,
        timeout_seconds: float | None = None,
    ) -> CommandExecution:
        started = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                input=input_text,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise EnvironmentError(f"executable not found: {argv[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            return CommandExecution(
                exit_code=None,
                stdout=_decode_output(exc.stdout),
                stderr=_decode_output(exc.stderr),
                duration_ms=round((time.monotonic() - started) * 1000, 3),
                timed_out=True,
            )
        return CommandExecution(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_ms=round((time.monotonic() - started) * 1000, 3),
        )


@dataclass(frozen=True)
class DockerTaskSpec:
    task_id: str
    image: str
    base_commit: str
    test_patch: str
    repository_path: str = "/testbed"
    fail_to_pass: tuple[str, ...] = ()
    pass_to_pass: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if not self.image.strip() or self.image.startswith("-"):
            raise ValueError("image must be a non-empty Docker image reference")
        if not self.base_commit.strip():
            raise ValueError("base_commit must not be empty")
        if not self.test_patch.strip():
            raise ValueError("test_patch must not be empty")
        if not self.fail_to_pass:
            raise ValueError("fail_to_pass must not be empty")
        path = PurePosixPath(self.repository_path)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("repository_path must be an absolute container path")


@dataclass(frozen=True)
class DockerSandboxConfig:
    docker_binary: str = "docker"
    network: str = "none"
    memory_limit: str = "2g"
    cpu_limit: float = 1.0
    pids_limit: int = 256
    platform: str = "linux/amd64"
    startup_timeout_seconds: float = 60.0
    command_timeout_seconds: float = 30.0
    test_timeout_seconds: float = 1800.0
    max_output_chars: int = 20_000

    def __post_init__(self) -> None:
        if not self.docker_binary.strip():
            raise ValueError("docker_binary must not be empty")
        if self.network != "none":
            raise ValueError("Docker sandboxes must disable networking")
        if not self.memory_limit.strip():
            raise ValueError("memory_limit must not be empty")
        if self.cpu_limit <= 0 or self.pids_limit <= 0:
            raise ValueError("CPU and PID limits must be positive")
        if not self.platform.strip():
            raise ValueError("platform must not be empty")
        if min(self.startup_timeout_seconds, self.command_timeout_seconds, self.test_timeout_seconds) <= 0:
            raise ValueError("timeouts must be positive")
        if self.max_output_chars <= 0:
            raise ValueError("max_output_chars must be positive")

    def run_argv(self, image: str, command: tuple[str, ...]) -> tuple[str, ...]:
        """Build the constrained foreground form used for boundary inspection/tests."""

        if not image.strip() or image.startswith("-"):
            raise ValueError("image must be a non-empty Docker image reference")
        if not command:
            raise ValueError("command must not be empty")
        return (*self._security_argv(), image, *command)

    def start_argv(self, container_name: str, spec: DockerTaskSpec) -> tuple[str, ...]:
        return (
            *self._security_argv(),
            "--detach",
            "--name",
            container_name,
            "--workdir",
            spec.repository_path,
            spec.image,
            "sleep",
            "infinity",
        )

    def exec_argv(
        self,
        container_name: str,
        repository_path: str,
        command: tuple[str, ...],
        *,
        interactive: bool = False,
    ) -> tuple[str, ...]:
        prefix = (self.docker_binary, "exec", "-i") if interactive else (self.docker_binary, "exec")
        return (*prefix, "--workdir", repository_path, container_name, *command)

    def remove_argv(self, container_name: str) -> tuple[str, ...]:
        return (self.docker_binary, "rm", "--force", container_name)

    def _security_argv(self) -> tuple[str, ...]:
        return (
            self.docker_binary,
            "run",
            "--rm",
            "--platform",
            self.platform,
            "--network",
            self.network,
            "--memory",
            self.memory_limit,
            "--cpus",
            str(self.cpu_limit),
            "--pids-limit",
            str(self.pids_limit),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
        )


class DockerSandboxProvider:
    def __init__(
        self,
        task_specs: Mapping[str, DockerTaskSpec],
        config: DockerSandboxConfig | None = None,
        runner: CommandRunner | None = None,
    ) -> None:
        self.task_specs = dict(task_specs)
        self.config = config or DockerSandboxConfig()
        self.runner = runner or SubprocessCommandRunner()

    def create(self, task: CodingTask) -> CodingEnvironment:
        try:
            spec = self.task_specs[task.task_id]
        except KeyError as exc:
            raise EnvironmentError(f"no Docker environment spec for task {task.task_id}") from exc
        if spec.base_commit != task.base_commit:
            raise EnvironmentError(f"base commit mismatch for task {task.task_id}")
        return DockerSandboxEnvironment(spec, self.config, self.runner)


class DockerSandboxEnvironment:
    _LIST_FILES_SCRIPT = (
        "from pathlib import Path; "
        "root=Path('.'); "
        "print('\\n'.join(sorted(str(p) for p in root.rglob('*') "
        "if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts)))"
    )
    _READ_FILE_SCRIPT = """
from pathlib import Path
import sys

MAX_LINES = 200
MAX_CHARS = 8000

lines = Path(sys.argv[1]).read_text(encoding='utf-8').splitlines()
total = len(lines)
if total == 0:
    print('[file is empty]')
    raise SystemExit(0)
if len(sys.argv) == 4:
    first = max(1, int(sys.argv[2]))
    last = min(total, int(sys.argv[3]))
else:
    first = 1
    last = min(total, MAX_LINES)
if last < first:
    print(f'[no lines in range: the file has {total} lines]')
    raise SystemExit(0)
rendered = []
used_chars = 0
shown_last = first - 1
for number in range(first, last + 1):
    line = f'{number}: {lines[number - 1]}'
    if rendered and used_chars + len(line) + 1 > MAX_CHARS:
        break
    rendered.append(line)
    used_chars += len(line) + 1
    shown_last = number
if shown_last < total:
    notes = ['character budget reached'] if shown_last < last else []
    notes.append(f'file has {total} lines')
    notes.append(
        'continue with read_file start_line='
        f'{shown_last + 1} end_line={min(total, shown_last + MAX_LINES)}'
    )
    rendered.append(f'[read_file lines {first}-{shown_last}: ' + '; '.join(notes) + ']')
print('\\n'.join(rendered))
""".strip()
    _SEARCH_TEXT_SCRIPT = """
from pathlib import Path
import difflib
import re
import sys

TOTAL_LIMIT = 100
PER_FILE_LIMIT = 5
FALLBACK_LIMIT = 20
CANDIDATE_LIMIT = 6
RELATED_MATCH_LIMIT = 8
RELATED_FILE_LIMIT = 12
RELATED_MAX_CHARS = 2400
PATH_MATCH_PREFIX = 'PATH_MATCH:'
MAX_CHARS = 8000
DOCUMENTATION_DIRECTORIES = {'docs', 'doc', 'examples', 'example'}
DOCUMENTATION_NAMES = (
    'changelog', 'implementation_coverage', 'contributing', 'readme',
    'notice', 'license', 'authors', 'news',
)
IMPORT_STATEMENT = re.compile(
    r'^[ \\t]*(?:from[ \\t]+([A-Za-z_][\\w.]*)[ \\t]+import|import[ \\t]+([A-Za-z_][\\w.]*))',
    re.MULTILINE,
)


def location_rank(parts):
    if any(part in DOCUMENTATION_DIRECTORIES for part in parts):
        return 2
    name = parts[-1] if parts else ''
    if name.endswith(('.md', '.rst', '.txt')) or name.startswith(DOCUMENTATION_NAMES):
        return 2
    if any(part == 'tests' or part.startswith('test') for part in parts):
        return 1
    return 0


def listing():
    return tuple(
        sorted(
            path.relative_to(Path('.')).as_posix()
            for path in Path('.').rglob('*')
            if path.is_file()
            and not any(
                part in {'.git', '__pycache__', '.pytest_cache'}
                or part.endswith('.egg-info')
                for part in path.relative_to(Path('.')).parts
            )
        )
    )


def collect(query, files, per_file_limit):
    folded = query.casefold()
    ranked = []
    for relative in files:
        parts = tuple(part.casefold() for part in Path(relative).parts)
        rank = location_rank(parts)
        if folded in relative.casefold():
            ranked.append((rank, 0, relative, 0, f'{PATH_MATCH_PREFIX}{relative}'))
        path = Path(relative)
        try:
            if path.stat().st_size > 1_000_000:
                continue
            lines = path.read_text(encoding='utf-8').splitlines()
        except (OSError, UnicodeError):
            continue
        matched_in_file = 0
        for number, line in enumerate(lines, start=1):
            if folded not in line.casefold():
                continue
            ranked.append((rank, 1, relative, number, f'{relative}:{number}:{line[:300]}'))
            matched_in_file += 1
            if matched_in_file >= per_file_limit:
                break
    ranked.sort(key=lambda item: item[:4])
    return [(item[0], item[2], item[4]) for item in ranked]


def imported_modules(text, available):
    resolved = []
    for match in IMPORT_STATEMENT.finditer(text):
        module = match.group(1) or match.group(2)
        if not module:
            continue
        relative = module.replace('.', '/')
        for candidate in (f'{relative}.py', f'{relative}/__init__.py'):
            if candidate in available and candidate not in resolved:
                resolved.append(candidate)
                break
    return sorted(resolved, key=lambda path: (-path.count('/'), path.endswith('/__init__.py')))


def candidate_lines(matches, files):
    matched_files = []
    for rank, relative, _ in matches:
        if rank == 0:
            return []
        if relative not in matched_files:
            matched_files.append(relative)
    if not matched_files:
        return []
    available = set(files)
    sources = []
    for relative in matched_files[:3]:
        try:
            sources.append(Path(relative).read_text(encoding='utf-8'))
        except (OSError, UnicodeError):
            continue
    resolved = []
    for source in sources:
        for path in imported_modules(source, available):
            if path not in resolved:
                resolved.append(path)
    ranked = sorted(resolved, key=lambda path: (-path.count('/'), path.endswith('/__init__.py')))
    return [f'IMPLEMENTATION_CANDIDATE:{path}' for path in ranked[:CANDIDATE_LIMIT]]


def related_lines(query, files):
    segments = {
        piece.strip("`'\\" ")
        for piece in query.split('/')
        if len(piece.strip("`'\\" ")) >= 3
    }
    for segment in sorted(segments, key=len, reverse=True):
        if segment.casefold() == query.casefold():
            continue
        grouped = {}
        for rank, relative, line in collect(segment, files, PER_FILE_LIMIT):
            if rank == 0 and not line.startswith(PATH_MATCH_PREFIX):
                grouped.setdefault(relative, []).append(line)
        if not grouped or len(grouped) > RELATED_FILE_LIMIT:
            continue
        folded = segment.casefold().replace('-', '_')
        ranked = sorted(
            grouped.items(),
            key=lambda item: (folded not in item[0].casefold(), -len(item[1]), item[0]),
        )
        shown = []
        for index, (_, file_lines) in enumerate(ranked):
            if index >= 3:
                break
            shown.extend(file_lines[: 4 if index == 0 else 2])
        return [
            f'No implementation file contains "{query}". Shorter query "{segment}" matches '
            'implementation files:',
            *bounded(shown[:RELATED_MATCH_LIMIT], RELATED_MAX_CHARS),
        ]
    return []


def navigation_lines(query, matches, files):
    lines = []
    if not any(rank == 0 for rank, _, _ in matches):
        lines.extend(related_lines(query, files))
    lines.extend(candidate_lines(matches, files))
    return lines


def bounded(lines, max_chars):
    rendered = []
    used_chars = 0
    for line in lines:
        if rendered and used_chars + len(line) + 1 > max_chars:
            break
        rendered.append(line)
        used_chars += len(line) + 1
    if len(rendered) < len(lines):
        rendered.append(
            f'[search_text: showing {len(rendered)} of {len(lines)} matches; '
            'narrow the query or read the first file]'
        )
    return rendered


query = sys.argv[1]
files = listing()
matches = collect(query, files, PER_FILE_LIMIT)
if matches:
    shown = matches[:TOTAL_LIMIT]
    nav = navigation_lines(query, shown, files)
    reserved = sum(len(line) + 1 for line in nav)
    output = bounded([line for _, _, line in shown], max(MAX_CHARS // 2, MAX_CHARS - reserved))
    output.extend(nav)
else:
    output = [f'No exact matches for: {query}']
    # A phrase query is retried on its longest word, which is what reaches an identifier. A
    # path-like query needs the other treatment: the whole path is one whitespace token, so
    # retrying it repeats the miss, while its basename and its later segments are names the
    # listing can answer - `moto/config/server.py` does not exist and `server.py` does.
    if '/' in query:
        stripped = query.strip().strip('"\\'`')
        parts = [part for part in stripped.split('/') if part]
        attempts = []
        if parts:
            name = parts[-1]
            if len(name) >= 4:
                attempts.append((name, f'Basename of the query: {name}'))
            for part in reversed(parts[:-1]):
                if len(part) >= 4:
                    attempts.append((part, f'Path segment of the query: {part}'))
    else:
        tokens = sorted(
            {token for token in query.split() if len(token) >= 4}, key=len, reverse=True
        )
        attempts = [(token, f'Longest token in the query: {token}') for token in tokens[:1]]
    for token, label in attempts:
        token_matches = collect(token, files, PER_FILE_LIMIT)
        if token_matches:
            output.append(label)
            shown = token_matches[:FALLBACK_LIMIT]
            nav = navigation_lines(query, shown, files)
            reserved = sum(len(line) + 1 for line in nav)
            output.extend(
                bounded([line for _, _, line in shown], max(MAX_CHARS // 2, MAX_CHARS - reserved))
            )
            output.extend(nav)
            break
    else:
        candidates = {name.casefold(): name for name in files}
        suggestions = difflib.get_close_matches(query.casefold(), candidates, n=8, cutoff=0.45)
        output.extend(f'SUGGESTED_PATH:{candidates[item]}' for item in suggestions)
    output = bounded(output, MAX_CHARS)
print('\\n'.join(output))
""".strip()
    _REPLACE_TEXT_SCRIPT = """
from pathlib import Path
import ast
import sys


def collapse_whitespace(content):
    collapsed = []
    offsets = []
    after_whitespace = False
    for index, character in enumerate(content):
        if character.isspace():
            after_whitespace = True
            continue
        if collapsed and after_whitespace:
            collapsed.append(' ')
            offsets.append(index - 1)
        collapsed.append(character)
        offsets.append(index)
        after_whitespace = False
    return ''.join(collapsed), offsets


def occurrences(content, old):
    positions = []
    start = content.find(old)
    while start >= 0:
        positions.append(start)
        start = content.find(old, start + 1)
    return positions


def mismatch_message(content, old):
    found = content.count(old)
    if found > 1:
        lines = [content.count('\\n', 0, index) + 1 for index in occurrences(content, old)]
        listed = ', '.join(str(line) for line in lines[:8])
        return (
            f'replace_text requires exactly one match, found {found}: lines {listed}. '
            'Include more surrounding context, such as a whole line or two, so that `old` '
            'matches exactly once.'
        )
    collapsed_content, offsets = collapse_whitespace(content)
    collapsed_old = ' '.join(old.split())
    if collapsed_old:
        index = collapsed_content.find(collapsed_old)
        if index >= 0 and collapsed_content.find(collapsed_old, index + 1) < 0:
            start = offsets[index]
            end = offsets[index + len(collapsed_old) - 1] + 1
            first_line = content.count('\\n', 0, start) + 1
            last_line = content.count('\\n', 0, end) + 1
            exact = content[start:end]
            shown = exact if len(exact) <= 300 else exact[:300] + '...'
            return (
                'replace_text requires exactly one match, found 0. The same text appears at '
                f'lines {first_line}-{last_line} with different whitespace, so `old` has to '
                f'repeat the file byte for byte: {shown!r}. Use exactly that value, or '
                f'replace_lines start_line={first_line} end_line={last_line}.'
            )
    return (
        'replace_text requires exactly one match, found 0. Search results print one matching '
        'line at a time, so a value joined from two result lines never matches the file. Read '
        'the file and copy `old` from the numbered read_file output, or use replace_lines on '
        'the range that read shows. If this text came from a verifier failure or the issue, run '
        'search_text with it: the file that produces it is not necessarily the file you have '
        'open.'
    )


def enclosed_statement_span(original, replaced_lines):
    if replaced_lines is None:
        return None
    start_line, end_line = replaced_lines
    try:
        module = ast.parse(original)
    except SyntaxError:
        return None
    span = None
    for node in ast.walk(module):
        if not isinstance(node, ast.stmt):
            continue
        first, last = node.lineno, node.end_lineno
        if first is None or last is None:
            continue
        if (first, last) == (start_line, end_line):
            return None
        if first <= start_line and last >= end_line:
            if span is None or (last - first) < (span[1] - span[0]):
                span = (first, last)
    return span


def unparseable_message(path, original, replaced_lines, exc):
    message = (
        f'edit not applied: it would leave {path} unparseable '
        f'({type(exc).__name__}: {exc.msg} at line {exc.lineno}). Replace the whole '
        'statement, including its indentation, in one edit.'
    )
    span = enclosed_statement_span(original, replaced_lines)
    if span is not None:
        message += (
            f' The statement you replaced lines {replaced_lines[0]}-{replaced_lines[1]} of '
            f'spans lines {span[0]}-{span[1]}: give replace_lines that whole range.'
        )
    return message


path = Path(sys.argv[1])
old = sys.argv[2]
new = sys.argv[3]
content = path.read_text(encoding='utf-8')
if content.count(old) != 1:
    print(mismatch_message(content, old), file=sys.stderr)
    raise SystemExit(1)
updated = content.replace(old, new, 1)
if path.suffix == '.py':
    try:
        compile(updated, str(path), 'exec')
    except SyntaxError as exc:
        offset = content.find(old)
        first = content.count('\\n', 0, offset) + 1
        print(unparseable_message(path, content, (first, first + old.count('\\n')), exc), file=sys.stderr)
        raise SystemExit(1)
path.write_text(updated, encoding='utf-8')
""".strip()
    _REPLACE_LINES_SCRIPT = """
from pathlib import Path
import ast
import sys


def enclosed_statement_span(original, replaced_lines):
    if replaced_lines is None:
        return None
    start_line, end_line = replaced_lines
    try:
        module = ast.parse(original)
    except SyntaxError:
        return None
    span = None
    for node in ast.walk(module):
        if not isinstance(node, ast.stmt):
            continue
        first, last = node.lineno, node.end_lineno
        if first is None or last is None:
            continue
        if (first, last) == (start_line, end_line):
            return None
        if first <= start_line and last >= end_line:
            if span is None or (last - first) < (span[1] - span[0]):
                span = (first, last)
    return span


def unparseable_message(path, original, replaced_lines, exc):
    message = (
        f'edit not applied: it would leave {path} unparseable '
        f'({type(exc).__name__}: {exc.msg} at line {exc.lineno}). Replace the whole '
        'statement, including its indentation, in one edit.'
    )
    span = enclosed_statement_span(original, replaced_lines)
    if span is not None:
        message += (
            f' The statement you replaced lines {replaced_lines[0]}-{replaced_lines[1]} of '
            f'spans lines {span[0]}-{span[1]}: give replace_lines that whole range.'
        )
    return message


path = Path(sys.argv[1])
start_line = int(sys.argv[2])
end_line = int(sys.argv[3])
new = sys.argv[4]
content = path.read_text(encoding='utf-8')
lines = content.splitlines(keepends=True)
if end_line > len(lines):
    raise ValueError(f'replace_lines end_line {end_line} exceeds file length {len(lines)}')
selected = ''.join(lines[start_line - 1:end_line])
if new and selected.endswith('\\n') and not new.endswith(('\\n', '\\r')):
    new += '\\n'
updated = ''.join(lines[:start_line - 1]) + new + ''.join(lines[end_line:])
if updated == content:
    print(
        f'replace_lines would not change {path}: the replacement is identical to the lines it '
        'replaces. This edit is refused instead of reported as applied, so change the code those '
        'lines contain.',
        file=sys.stderr,
    )
    raise SystemExit(1)
if path.suffix == '.py':
    try:
        compile(updated, str(path), 'exec')
    except SyntaxError as exc:
        print(
            unparseable_message(path, content, (start_line, end_line), exc),
            file=sys.stderr,
        )
        raise SystemExit(1)
path.write_text(updated, encoding='utf-8')
""".strip()
    _PATCH_VALID_SCRIPT = """
from pathlib import Path
import sys

for raw in sys.argv[1:]:
    path = Path(raw)
    if not path.is_file():
        raise SystemExit(1)
    if path.suffix == '.py':
        compile(path.read_text(encoding='utf-8'), raw, 'exec')
""".strip()

    def __init__(self, spec: DockerTaskSpec, config: DockerSandboxConfig, runner: CommandRunner) -> None:
        self.spec = spec
        self.config = config
        self.runner = runner
        self.task: CodingTask | None = None
        self.container_name: str | None = None
        self._container_started = False
        self.baseline_result: TestResult | None = None
        self.last_test_result: TestResult | None = None
        self.steps = 0
        self.tool_calls = 0
        self.violations: list[str] = []
        self._changed_files: set[str] = set()
        self._protected_files: set[str] = set()
        self._read_spans: dict[str, list[tuple[int, int]]] = {}
        self._edited_since_verification = False
        self._action_loop_guard = ActionLoopGuard(
            lambda path: path in self._protected_files or is_test_path(path)
        )

    def reset(self, task: CodingTask) -> str:
        self.close()
        if task.task_id != self.spec.task_id:
            raise EnvironmentError("Docker environment received the wrong task")
        self.task = task
        self.container_name = _container_name(task.task_id)
        self.steps = 0
        self.tool_calls = 0
        self.violations = []
        self._read_spans = {}
        self._action_loop_guard.reset()
        try:
            started = self.runner.run(
                self.config.start_argv(self.container_name, self.spec),
                timeout_seconds=self.config.startup_timeout_seconds,
            )
            if not started.passed:
                self._fail_and_close("failed to start Docker container", started)
            self._container_started = True
            head = self._exec(("git", "rev-parse", "HEAD"))
            if not head.passed or head.stdout.strip() != self.spec.base_commit:
                raise EnvironmentError(
                    f"container base commit mismatch: expected {self.spec.base_commit}, got {head.stdout.strip() or 'unknown'}"
                )
            patched = self._exec(
                ("git", "apply", "--whitespace=nowarn", "-"),
                input_text=self.spec.test_patch,
                interactive=True,
            )
            if not patched.passed:
                raise EnvironmentError(f"failed to apply SWE-Gym test patch: {patched.stderr or patched.stdout}")
            self._protected_files = _paths_from_patch(self.spec.test_patch)
            if not self._protected_files:
                raise EnvironmentError("SWE-Gym test patch does not declare any protected files")
            self.baseline_result = self._run_tests()
            self.last_test_result = self.baseline_result
            if self.baseline_result.passed:
                raise EnvironmentError(f"task {task.task_id} is invalid: baseline tests already pass")
        except Exception:
            self.close()
            raise
        return f"Baseline verifier result:\n{self._test_observation(self.baseline_result)}"

    def step(self, action: AgentAction) -> StepResult:
        task = self._require_active()
        if self.steps >= task.max_steps:
            return StepResult("Step budget exhausted.", True, self.last_test_result)
        self.steps += 1
        self.tool_calls += 1
        rejection = self._action_loop_guard.rejection_for(action)
        if rejection:
            remaining_steps = task.max_steps - self.steps
            result = StepResult(
                f"Tool error: {rejection} "
                f"You have {remaining_steps} tool steps left; switch to a different tool or target, "
                "and prioritize an evidence-backed source edit when enough context is available.",
                False,
                self.last_test_result,
            )
            self._action_loop_guard.record(action, result.observation)
            return result
        try:
            if action.kind is ActionKind.LIST_FILES:
                result = self._exec(("python", "-c", self._LIST_FILES_SCRIPT))
                self._require_command(result, "list_files")
                step_result = StepResult(self._bounded_listing(result.stdout), False)
            elif action.kind is ActionKind.SEARCH_TEXT:
                query = self._required_string(action.arguments, "query")
                if len(query) > 200:
                    raise ToolError("search_text query must be at most 200 characters")
                result = self._exec(("python", "-c", self._SEARCH_TEXT_SCRIPT, query))
                self._require_command(result, "search_text")
                step_result = StepResult(
                    self._bounded_observation(
                        result.stdout,
                        note="narrow the search_text query or read the first file",
                    ),
                    False,
                )
            elif action.kind is ActionKind.READ_FILE:
                path = self._safe_relative_path(action.arguments.get("path"))
                line_range = read_line_range(action.arguments)
                command = ("python", "-c", self._READ_FILE_SCRIPT, path)
                if line_range is not None:
                    command = (*command, str(line_range[0]), str(line_range[1]))
                result = self._exec(command)
                self._require_command(result, "read_file")
                observation = self._bounded_observation(
                    result.stdout,
                    note="read a smaller line range to see the rest of the file",
                )
                span = shown_line_span(observation)
                if span is not None:
                    self._read_spans.setdefault(path, []).append(span)
                step_result = StepResult(observation, False)
            elif action.kind is ActionKind.REPLACE_TEXT:
                path = self._safe_relative_path(action.arguments.get("path"))
                if path in self._protected_files:
                    raise EnvironmentError(f"cannot modify verifier-owned test file: {path}")
                old = self._required_string(action.arguments, "old")
                new = self._required_string(action.arguments, "new", allow_empty=True)
                if old == new:
                    raise ToolError(
                        f"replace_text would not change {path}: `new` is identical to `old`. "
                        "This edit is refused instead of reported as applied, so pick the "
                        "statement that produces the failing value and replace it with corrected "
                        "code, or use replace_lines on the range read_file showed."
                    )
                result = self._exec(("python", "-c", self._REPLACE_TEXT_SCRIPT, path, old, new))
                self._require_command(result, "replace_text")
                self._changed_files.add(path)
                self._read_spans.pop(path, None)
                self._edited_since_verification = True
                step_result = StepResult(f"Updated {path}.", False)
            elif action.kind is ActionKind.REPLACE_LINES:
                path = self._safe_relative_path(action.arguments.get("path"))
                if path in self._protected_files:
                    raise EnvironmentError(f"cannot modify verifier-owned test file: {path}")
                spans = self._read_spans.get(path)
                if not spans:
                    raise ToolError("replace_lines requires reading the target file first")
                start_line, end_line = replace_line_range_bounds(
                    action.arguments.get("start_line"), action.arguments.get("end_line")
                )
                if not any(first <= start_line and end_line <= last for first, last in spans):
                    raise ToolError(
                        read_range_requirement_message(path, start_line, end_line, spans)
                    )
                new = self._required_string(action.arguments, "new", allow_empty=True)
                result = self._exec(
                    (
                        "python",
                        "-c",
                        self._REPLACE_LINES_SCRIPT,
                        path,
                        str(start_line),
                        str(end_line),
                        new,
                    )
                )
                self._require_command(result, "replace_lines")
                self._changed_files.add(path)
                self._read_spans.pop(path, None)
                self._edited_since_verification = True
                step_result = StepResult(f"Updated {path}.", False)
            elif action.kind is ActionKind.RUN_TESTS:
                result = self._run_tests()
                self.last_test_result = result
                self._edited_since_verification = False
                step_result = StepResult(
                    self._test_observation(result),
                    result.passed or result.timed_out,
                    result,
                )
            elif action.kind is ActionKind.FINISH:
                # `finish` re-runs the verifier, so a finish with no edit behind it can be
                # refused with the failure the policy needs instead of ending the episode.
                # An unrepaired failure stays valid evidence, so it is reused rather than
                # paid for twice. A finish that follows an edit is the one case that has to
                # run: nothing else can change a failure, and a policy that probes with
                # `finish` between reads otherwise pays for a container run per probe.
                verified = self.last_test_result
                if verified is None or verified.passed or self._edited_since_verification:
                    verified = self._run_tests()
                    self.last_test_result = verified
                    self._edited_since_verification = False
                refusal = premature_finish_refusal(
                    self._action_loop_guard.patched,
                    verified,
                    steps_remaining=task.max_steps - self.steps,
                )
                if refusal is not None:
                    step_result = StepResult(
                        f"Tool error: {refusal}\n{self._test_observation(verified)}",
                        False,
                        verified,
                    )
                else:
                    step_result = StepResult(self._test_observation(verified), True, verified)
            else:
                raise EnvironmentError(f"unsupported action: {action.kind.value}")
        except (ToolError, UnicodeError) as exc:
            step_result = StepResult(f"Tool error: {exc}", False, self.last_test_result)
        except EnvironmentError as exc:
            violation = action_violation_code(action, exc)
            self.violations.append(violation)
            step_result = StepResult(str(exc), True, self.last_test_result, violation)
        self._action_loop_guard.record(action, step_result.observation)
        return step_result

    def finalize(self) -> TestResult:
        self._require_active()
        result = self._run_tests()
        self.last_test_result = result
        self._edited_since_verification = False
        return result

    def changed_files(self) -> tuple[str, ...]:
        return tuple(sorted(self._changed_files))

    def patch_is_valid(self) -> bool:
        changed = self.changed_files()
        if not changed:
            return False
        result = self._exec(("python", "-c", self._PATCH_VALID_SCRIPT, *changed))
        return result.passed

    def graded_targets(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The declared SWE-Gym FAIL_TO_PASS and PASS_TO_PASS nodes for this task."""

        return self.spec.fail_to_pass, self.spec.pass_to_pass

    @property
    def loop_rejections(self) -> int:
        """Count of policy actions the loop guard refused as unproductive repeats."""

        return self._action_loop_guard.rejection_count

    def close(self) -> None:
        container_name = self.container_name
        self.container_name = None
        if container_name is not None and self._container_started:
            self.runner.run(
                self.config.remove_argv(container_name),
                timeout_seconds=self.config.startup_timeout_seconds,
            )
        self._container_started = False
        self.task = None
        self.baseline_result = None
        self.last_test_result = None
        self._changed_files = set()
        self._protected_files = set()
        self._read_spans = {}
        self._edited_since_verification = False
        self._action_loop_guard.reset()

    def __enter__(self) -> DockerSandboxEnvironment:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _run_tests(self) -> TestResult:
        task = self._require_active()
        execution = self._exec(task.test_command, timeout_seconds=self.config.test_timeout_seconds)
        return TestResult(
            command=task.test_command,
            passed=execution.passed,
            exit_code=execution.exit_code,
            stdout=execution.stdout[-self.config.max_output_chars :],
            stderr=execution.stderr[-self.config.max_output_chars :],
            duration_ms=execution.duration_ms,
            timed_out=execution.timed_out,
        )

    def _exec(
        self,
        command: tuple[str, ...],
        *,
        input_text: str | None = None,
        interactive: bool = False,
        timeout_seconds: float | None = None,
    ) -> CommandExecution:
        if self.container_name is None:
            raise EnvironmentError("Docker environment is not active")
        return self.runner.run(
            self.config.exec_argv(
                self.container_name,
                self.spec.repository_path,
                command,
                interactive=interactive,
            ),
            input_text=input_text,
            timeout_seconds=timeout_seconds or self.config.command_timeout_seconds,
        )

    def _fail_and_close(self, message: str, result: CommandExecution) -> None:
        detail = result.stderr or result.stdout
        self.close()
        raise EnvironmentError(f"{message}: {detail.strip() or 'unknown Docker error'}")

    def _require_active(self) -> CodingTask:
        if self.task is None or self.container_name is None:
            raise EnvironmentError("Docker environment is not active")
        return self.task

    @staticmethod
    def _safe_relative_path(raw: Any) -> str:
        if not isinstance(raw, str) or not raw:
            raise ToolError("path must be a non-empty string")
        path = PurePosixPath(raw.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise EnvironmentError("path escapes repository workspace")
        return str(path)

    @staticmethod
    def _required_string(arguments: dict[str, Any], name: str, *, allow_empty: bool = False) -> str:
        value = arguments.get(name)
        if not isinstance(value, str) or (not allow_empty and not value):
            raise ToolError(f"{name} must be a {'string' if allow_empty else 'non-empty string'}")
        return value

    @staticmethod
    def _require_command(result: CommandExecution, action: str) -> None:
        if not result.passed:
            # The container scripts already phrase a refused no-op as a policy instruction; the
            # generic `failed: <stderr>` wrapper would bury it behind a subprocess step.
            detail = result.stderr or result.stdout
            if "would not change" in detail:
                raise ToolError(detail.strip())
            raise ToolError(f"{action} failed: {detail}")

    def _bounded_listing(self, content: str) -> str:
        return self._bounded_observation(
            content,
            note="use search_text or run_tests to locate relevant code",
        )

    def _bounded_observation(self, content: str, *, note: str) -> str:
        """Keep the head and tail of an observation, and name why it was cut.

        The head carries the highest-ranked matches and the first numbered lines, and
        the tail carries `read_file`'s next-step footer, so neither end may be dropped.
        """

        if len(content) <= self.config.max_output_chars:
            return content
        marker = f"\n...[observation truncated; {note}]...\n"
        half = (self.config.max_output_chars - len(marker)) // 2
        return content[:half] + marker + content[-half:]

    @staticmethod
    def _test_observation(result: TestResult) -> str:
        status = "passed" if result.passed else "failed"
        summary = failure_summary(result)
        detail = verifier_output_detail(result)
        if summary:
            return f"Tests {status} (exit={result.exit_code}).\n{summary}\n{detail}"
        return f"Tests {status} (exit={result.exit_code}).\n{detail}"


def _container_name(task_id: str) -> str:
    safe = "".join(character if character.isalnum() or character in "_.-" else "-" for character in task_id)
    return f"coding-agent-{safe[:40]}-{uuid.uuid4().hex[:8]}".lower()


def _paths_from_patch(patch: str) -> set[str]:
    paths: set[str] = set()
    for line in patch.splitlines():
        if not line.startswith(("--- a/", "+++ b/")):
            continue
        path = line[6:].split("\t", 1)[0]
        if path and path != "/dev/null":
            paths.add(path)
    return paths


def _decode_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
