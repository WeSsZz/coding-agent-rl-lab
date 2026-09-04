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
    SEARCH_TEXT = "search_text"
    READ_FILE = "read_file"
    REPLACE_TEXT = "replace_text"
    REPLACE_LINES = "replace_lines"
    RUN_TESTS = "run_tests"
    FINISH = "finish"


@dataclass(frozen=True)
class CodingTask:
    task_id: str
    issue: str
    fixture_path: str | None
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
class PolicyDecision:
    action: AgentAction
    input_messages: tuple[dict[str, str], ...] = ()
    output_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    violation: str | None = None


@dataclass(frozen=True)
class TestResult:
    command: tuple[str, ...]
    passed: bool
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TestResult:
        return cls(
            command=tuple(value["command"]),
            passed=bool(value["passed"]),
            exit_code=value.get("exit_code"),
            stdout=str(value.get("stdout", "")),
            stderr=str(value.get("stderr", "")),
            duration_ms=float(value.get("duration_ms", 0.0)),
            timed_out=bool(value.get("timed_out", False)),
        )


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
    policy_input: tuple[dict[str, str], ...] = ()
    policy_output: str | None = None
    policy_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrajectoryStep:
        raw_test_result = value.get("test_result")
        return cls(
            sequence=int(value["sequence"]),
            action=AgentAction.from_dict(value["action"]),
            observation=str(value.get("observation", "")),
            terminated=bool(value.get("terminated", False)),
            test_result=(
                TestResult.from_dict(raw_test_result)
                if isinstance(raw_test_result, dict)
                else None
            ),
            violation=value.get("violation"),
            policy_input=tuple(dict(message) for message in value.get("policy_input", ())),
            policy_output=value.get("policy_output"),
            policy_metadata=dict(value.get("policy_metadata", {})),
        )


@dataclass(frozen=True)
class RewardVector:
    task_success: bool
    tests_passed: bool
    regression_free: bool
    patch_created: bool
    tool_calls: int
    steps: int
    violations: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RewardVector:
        return cls(
            task_success=bool(value["task_success"]),
            tests_passed=bool(value["tests_passed"]),
            regression_free=bool(value["regression_free"]),
            patch_created=bool(value["patch_created"]),
            tool_calls=int(value["tool_calls"]),
            steps=int(value["steps"]),
            violations=tuple(value.get("violations", ())),
        )

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

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PolicyManifest:
        return cls(
            policy_id=str(value["policy_id"]),
            version=str(value["version"]),
            policy_type=str(value["policy_type"]),
            model=value.get("model"),
            training_dataset=value.get("training_dataset"),
            parent_version=value.get("parent_version"),
            metadata=dict(value.get("metadata", {})),
        )


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
    initial_observation: str
    schema_version: int = 3

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for step in payload["steps"]:
            step["action"]["kind"] = step["action"]["kind"].value
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Trajectory:
        schema_version = int(value.get("schema_version", 0))
        if schema_version != 3:
            raise ValueError(f"unsupported trajectory schema version: {schema_version}")
        return cls(
            trajectory_id=str(value["trajectory_id"]),
            task_id=str(value["task_id"]),
            repetition=int(value["repetition"]),
            seed=int(value["seed"]),
            policy=PolicyManifest.from_dict(value["policy"]),
            steps=tuple(TrajectoryStep.from_dict(step) for step in value.get("steps", ())),
            reward=RewardVector.from_dict(value["reward"]),
            changed_files=tuple(value.get("changed_files", ())),
            baseline_tests_passed=bool(value["baseline_tests_passed"]),
            final_tests_passed=bool(value["final_tests_passed"]),
            initial_observation=str(value.get("initial_observation", "")),
            schema_version=schema_version,
        )
