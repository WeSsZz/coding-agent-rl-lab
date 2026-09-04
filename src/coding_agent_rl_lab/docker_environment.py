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
    read_line_range,
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

content = Path(sys.argv[1]).read_text(encoding='utf-8')
if len(sys.argv) == 4:
    start_line = int(sys.argv[2])
    end_line = int(sys.argv[3])
    content = ''.join(content.splitlines(keepends=True)[start_line - 1:end_line])
print(content, end='')
""".strip()
    _SEARCH_TEXT_SCRIPT = """
from pathlib import Path
import sys

query = sys.argv[1].casefold()
matches = []
for path in Path('.').rglob('*'):
    if not path.is_file() or any(part in {'.git', '__pycache__', '.pytest_cache'} for part in path.parts):
        continue
    relative = path.as_posix()
    parts = {part.casefold() for part in path.parts}
    if parts & {'docs', 'doc', 'examples', 'example'}:
        location_rank = 2
    elif any(part == 'tests' or part.startswith('test') for part in parts):
        location_rank = 1
    else:
        location_rank = 0
    if query in relative.casefold():
        matches.append((location_rank, 0, relative, 0, relative))
    try:
        too_large = path.stat().st_size > 1_000_000
    except OSError:
        continue
    if too_large:
        continue
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except (OSError, UnicodeError):
        continue
    for line_number, line in enumerate(lines, start=1):
        if query in line.casefold():
            rendered = f'{relative}:{line_number}:{line[:300]}'
            matches.append((location_rank, 1, relative, line_number, rendered))
matches.sort(key=lambda item: item[:4])
rendered_matches = [item[4] for item in matches[:100]]
print('\\n'.join(rendered_matches) if rendered_matches else f'No matches for: {sys.argv[1]}')
""".strip()
    _REPLACE_TEXT_SCRIPT = (
        "from pathlib import Path; import sys; "
        "p=Path(sys.argv[1]); old=sys.argv[2]; new=sys.argv[3]; "
        "s=p.read_text(encoding='utf-8'); n=s.count(old); "
        "assert n == 1, f'replace_text requires exactly one match, found {n}'; "
        "p.write_text(s.replace(old,new,1), encoding='utf-8')"
    )

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
        self._action_loop_guard = ActionLoopGuard()

    def reset(self, task: CodingTask) -> str:
        self.close()
        if task.task_id != self.spec.task_id:
            raise EnvironmentError("Docker environment received the wrong task")
        self.task = task
        self.container_name = _container_name(task.task_id)
        self.steps = 0
        self.tool_calls = 0
        self.violations = []
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
                step_result = StepResult(result.stdout[-self.config.max_output_chars :], False)
            elif action.kind is ActionKind.READ_FILE:
                path = self._safe_relative_path(action.arguments.get("path"))
                line_range = read_line_range(action.arguments)
                command = ("python", "-c", self._READ_FILE_SCRIPT, path)
                if line_range is not None:
                    command = (*command, str(line_range[0]), str(line_range[1]))
                result = self._exec(command)
                self._require_command(result, "read_file")
                step_result = StepResult(result.stdout[-self.config.max_output_chars :], False)
            elif action.kind is ActionKind.REPLACE_TEXT:
                path = self._safe_relative_path(action.arguments.get("path"))
                if path in self._protected_files:
                    raise EnvironmentError(f"cannot modify verifier-owned test file: {path}")
                old = self._required_string(action.arguments, "old")
                new = self._required_string(action.arguments, "new", allow_empty=True)
                result = self._exec(("python", "-c", self._REPLACE_TEXT_SCRIPT, path, old, new))
                self._require_command(result, "replace_text")
                self._changed_files.add(path)
                step_result = StepResult(f"Updated {path}.", False)
            elif action.kind is ActionKind.RUN_TESTS:
                result = self._run_tests()
                self.last_test_result = result
                step_result = StepResult(
                    self._test_observation(result),
                    result.passed or result.timed_out,
                    result,
                )
            elif action.kind is ActionKind.FINISH:
                result = self._run_tests()
                self.last_test_result = result
                step_result = StepResult(self._test_observation(result), True, result)
            else:
                raise EnvironmentError(f"unsupported action: {action.kind.value}")
        except (ToolError, UnicodeError) as exc:
            step_result = StepResult(f"Tool error: {exc}", False, self.last_test_result)
        except EnvironmentError as exc:
            violation = f"invalid_action:{type(exc).__name__}"
            self.violations.append(violation)
            step_result = StepResult(str(exc), True, self.last_test_result, violation)
        self._action_loop_guard.record(action, step_result.observation)
        return step_result

    def finalize(self) -> TestResult:
        self._require_active()
        result = self._run_tests()
        self.last_test_result = result
        return result

    def changed_files(self) -> tuple[str, ...]:
        return tuple(sorted(self._changed_files))

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
            raise EnvironmentError("path must be a non-empty string")
        path = PurePosixPath(raw.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise EnvironmentError("path escapes repository workspace")
        return str(path)

    @staticmethod
    def _required_string(arguments: dict[str, Any], name: str, *, allow_empty: bool = False) -> str:
        value = arguments.get(name)
        if not isinstance(value, str) or (not allow_empty and not value):
            raise EnvironmentError(f"{name} must be a {'string' if allow_empty else 'non-empty string'}")
        return value

    @staticmethod
    def _require_command(result: CommandExecution, action: str) -> None:
        if not result.passed:
            raise ToolError(f"{action} failed: {result.stderr or result.stdout}")

    def _bounded_listing(self, content: str) -> str:
        if len(content) <= self.config.max_output_chars:
            return content
        marker = "\n...[file list truncated; use search_text or run_tests to locate relevant code]...\n"
        half = (self.config.max_output_chars - len(marker)) // 2
        return content[:half] + marker + content[-half:]

    @staticmethod
    def _test_observation(result: TestResult) -> str:
        status = "passed" if result.passed else "failed"
        detail = result.stderr or result.stdout
        return f"Tests {status} (exit={result.exit_code}).\n{detail[-4000:]}"


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
