from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from .contracts import CodingTask
from .docker_environment import (
    CommandExecution,
    CommandRunner,
    DockerSandboxConfig,
    DockerSandboxEnvironment,
    DockerSandboxProvider,
    DockerTaskSpec,
    SubprocessCommandRunner,
)
from .environment import CodingEnvironment, LocalFixtureEnvironment
from .verifier import LocalPythonVerifier


@runtime_checkable
class EnvironmentProvider(Protocol):
    """Creates an isolated environment appropriate for one coding task."""

    def create(self, task: CodingTask) -> CodingEnvironment: ...


class LocalFixtureEnvironmentProvider:
    """Provider for trusted, repository-owned M0 fixtures only."""

    def __init__(self, project_root: str | Path, verifier: LocalPythonVerifier | None = None) -> None:
        self.project_root = Path(project_root).resolve()
        self.verifier = verifier

    def create(self, task: CodingTask) -> CodingEnvironment:
        del task
        return LocalFixtureEnvironment(self.project_root, self.verifier)

__all__ = [
    "CommandExecution",
    "CommandRunner",
    "DockerSandboxConfig",
    "DockerSandboxEnvironment",
    "DockerSandboxProvider",
    "DockerTaskSpec",
    "EnvironmentProvider",
    "LocalFixtureEnvironmentProvider",
    "SubprocessCommandRunner",
]
