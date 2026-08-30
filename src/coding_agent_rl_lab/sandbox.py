from __future__ import annotations

import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


class SandboxError(RuntimeError):
    pass


class DockerUnavailableError(SandboxError):
    pass


@dataclass(frozen=True)
class SandboxSpec:
    image: str
    workspace: Path
    command: tuple[str, ...]
    timeout_seconds: float = 300.0
    memory: str = "4g"
    cpus: float = 2.0
    pids_limit: int = 256
    environment: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _is_pinned_image(self.image):
            raise ValueError("sandbox image must use an immutable digest or an explicit non-latest tag")
        if not self.command or not all(isinstance(part, str) and part for part in self.command):
            raise ValueError("sandbox command must be a non-empty argv tuple")
        if self.timeout_seconds <= 0 or self.cpus <= 0 or self.pids_limit <= 0:
            raise ValueError("sandbox resource limits must be positive")
        if not re.fullmatch(r"[1-9][0-9]*(?:[kKmMgG])?", self.memory):
            raise ValueError("sandbox memory must look like 512m or 4g")
        for key, value in self.environment.items():
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key) or "SECRET" in key or "TOKEN" in key or "KEY" in key:
                raise ValueError(f"sandbox environment key is not allowed: {key}")
            if not isinstance(value, str):
                raise ValueError("sandbox environment values must be strings")


@dataclass(frozen=True)
class SandboxResult:
    command: tuple[str, ...]
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class DockerSandboxProvider:
    """Runs pre-pulled images with no network and a minimal Linux capability set."""

    def __init__(self, *, docker_executable: str = "docker", max_output_chars: int = 50_000) -> None:
        self.docker_executable = docker_executable
        self.max_output_chars = max_output_chars

    def available(self) -> bool:
        if shutil.which(self.docker_executable) is None:
            return False
        completed = subprocess.run(
            (self.docker_executable, "info"),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return completed.returncode == 0

    def build_command(self, spec: SandboxSpec) -> tuple[str, ...]:
        workspace = spec.workspace.resolve()
        if not workspace.is_dir():
            raise SandboxError(f"workspace does not exist: {workspace}")
        name = f"coding-agent-{uuid.uuid4().hex[:12]}"
        command = [
            self.docker_executable,
            "run",
            "--rm",
            "--pull",
            "never",
            "--name",
            name,
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(spec.pids_limit),
            "--memory",
            spec.memory,
            "--cpus",
            str(spec.cpus),
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
            "--volume",
            f"{workspace}:/workspace:rw",
            "--workdir",
            "/workspace",
        ]
        for key, value in sorted(spec.environment.items()):
            command.extend(("--env", f"{key}={value}"))
        command.extend((spec.image, *spec.command))
        return tuple(command)

    def run(self, spec: SandboxSpec) -> SandboxResult:
        if not self.available():
            raise DockerUnavailableError("Docker is not installed or the daemon is unavailable")
        command = self.build_command(spec)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=spec.timeout_seconds,
                check=False,
            )
            return SandboxResult(
                command=command,
                exit_code=completed.returncode,
                stdout=completed.stdout[-self.max_output_chars :],
                stderr=completed.stderr[-self.max_output_chars :],
                duration_ms=round((time.monotonic() - started) * 1000, 3),
            )
        except subprocess.TimeoutExpired as exc:
            subprocess.run(
                (self.docker_executable, "rm", "-f", _container_name(command)),
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            return SandboxResult(
                command=command,
                exit_code=None,
                stdout=_decode(exc.stdout)[-self.max_output_chars :],
                stderr=_decode(exc.stderr)[-self.max_output_chars :],
                duration_ms=round((time.monotonic() - started) * 1000, 3),
                timed_out=True,
            )


def _is_pinned_image(image: str) -> bool:
    if not image or any(character.isspace() for character in image):
        return False
    if "@sha256:" in image:
        digest = image.rsplit("@sha256:", 1)[1]
        return bool(re.fullmatch(r"[0-9a-f]{64}", digest))
    last_segment = image.rsplit("/", 1)[-1]
    if ":" not in last_segment:
        return False
    tag = last_segment.rsplit(":", 1)[1]
    return bool(tag and tag != "latest")


def _container_name(command: tuple[str, ...]) -> str:
    return command[command.index("--name") + 1]


def _decode(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
