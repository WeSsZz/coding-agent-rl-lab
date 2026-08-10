from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from .contracts import TestResult


class VerifierError(RuntimeError):
    pass


class LocalPythonVerifier:
    """Runs only trusted, argv-based Python tests for repository-owned fixtures."""

    def __init__(self, *, timeout_seconds: float = 5.0, max_output_chars: int = 20_000) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

    def run(self, repository: Path, command: tuple[str, ...]) -> TestResult:
        if not repository.is_dir():
            raise VerifierError(f"repository does not exist: {repository}")
        resolved = tuple(sys.executable if part == "{python}" else part for part in command)
        executable = Path(resolved[0]).name
        if not executable.startswith("python"):
            raise VerifierError(f"local verifier rejects executable: {executable}")
        started = time.monotonic()
        try:
            completed = subprocess.run(
                resolved,
                cwd=repository,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            duration_ms = (time.monotonic() - started) * 1000
            return TestResult(
                command=resolved,
                passed=completed.returncode == 0,
                exit_code=completed.returncode,
                stdout=completed.stdout[-self.max_output_chars :],
                stderr=completed.stderr[-self.max_output_chars :],
                duration_ms=round(duration_ms, 3),
            )
        except subprocess.TimeoutExpired as exc:
            duration_ms = (time.monotonic() - started) * 1000
            return TestResult(
                command=resolved,
                passed=False,
                exit_code=None,
                stdout=_decode_output(exc.stdout)[-self.max_output_chars :],
                stderr=_decode_output(exc.stderr)[-self.max_output_chars :],
                duration_ms=round(duration_ms, 3),
                timed_out=True,
            )


def _decode_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
