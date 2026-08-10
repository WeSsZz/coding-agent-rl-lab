from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class DatasetSplit(str, Enum):
    DEVELOPMENT = "development"
    REGRESSION = "regression"
    HELD_OUT = "held_out"


class ActionKind(str, Enum):
    LIST_FILES = "list_files"
    READ_FILE = "read_file"
    REPLACE_TEXT = "replace_text"
    RUN_TESTS = "run_tests"
    FINISH = "finish"


@dataclass(frozen=True)
class CodingTask:
    task_id: str
    issue: str
    fixture_path: str
    base_commit: str
    test_command: tuple[str, ...]
    split: DatasetSplit
    provenance: str
    max_steps: int = 8
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if not self.issue.strip():
            raise ValueError("issue must not be empty")
        if not self.test_command:
            raise ValueError("test_command must not be empty")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")


@dataclass(frozen=True)
class AgentAction:
    kind: ActionKind
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AgentAction:
        return cls(ActionKind(value["kind"]), dict(value.get("arguments", {})))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "arguments": self.arguments}


@dataclass(frozen=True)
class TestResult:
    command: tuple[str, ...]
    passed: bool
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False


@dataclass(frozen=True)
class StepResult:
    observation: str
    terminated: bool
    test_result: TestResult | None = None
    violation: str | None = None


@dataclass(frozen=True)
class TrajectoryStep:
    sequence: int
    action: AgentAction
    observation: str
    terminated: bool
    test_result: TestResult | None = None
    violation: str | None = None


@dataclass(frozen=True)
class RewardVector:
    task_success: bool
    tests_passed: bool
    regression_free: bool
    patch_created: bool
    tool_calls: int
    steps: int
    violations: tuple[str, ...] = ()

    @property
    def scalar(self) -> float:
        """A conservative adapter reward; structured components remain authoritative."""

        if self.violations or not self.task_success:
            return 0.0
        reward = 0.0
        reward += 0.65 if self.task_success else 0.0
        reward += 0.20 if self.regression_free else 0.0
        reward += 0.10 if self.patch_created else 0.0
        reward += max(0.0, 0.05 - 0.005 * self.tool_calls)
        return round(min(1.0, reward), 4)


@dataclass(frozen=True)
class PolicyManifest:
    policy_id: str
    version: str
    policy_type: str
    model: str | None = None
    training_dataset: str | None = None
    parent_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    task_id: str
    repetition: int
    seed: int
    policy: PolicyManifest
    steps: tuple[TrajectoryStep, ...]
    reward: RewardVector
    changed_files: tuple[str, ...]
    baseline_tests_passed: bool
    final_tests_passed: bool
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for step in payload["steps"]:
            step["action"]["kind"] = step["action"]["kind"].value
        return payload
