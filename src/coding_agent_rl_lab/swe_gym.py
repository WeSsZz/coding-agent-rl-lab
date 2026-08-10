from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class SWEGymAdapterError(ValueError):
    pass


@dataclass(frozen=True)
class SWEGymInstance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    test_patch: str = ""
    hints_text: str = ""
    version: str = ""
    image_name: str | None = None
    provenance: str = "SWE-Gym"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SWEGymInstance:
        try:
            instance = cls(
                instance_id=_required_string(value, "instance_id"),
                repo=_required_string(value, "repo"),
                base_commit=_required_string(value, "base_commit"),
                problem_statement=_required_string(value, "problem_statement"),
                fail_to_pass=_string_sequence(value.get("FAIL_TO_PASS", value.get("fail_to_pass", ()))),
                pass_to_pass=_string_sequence(value.get("PASS_TO_PASS", value.get("pass_to_pass", ()))),
                test_patch=str(value.get("test_patch", "")),
                hints_text=str(value.get("hints_text", "")),
                version=str(value.get("version", "")),
                image_name=_optional_string(value.get("image_name")),
                provenance=str(value.get("provenance", "SWE-Gym")),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SWEGymAdapterError(str(exc)) from exc
        if not instance.fail_to_pass:
            raise SWEGymAdapterError(f"{instance.instance_id}: FAIL_TO_PASS must not be empty")
        return instance

    def agent_input(self) -> dict[str, str]:
        """Return only fields the policy may see; gold patch and test patch stay hidden."""

        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "hints_text": self.hints_text,
        }


@dataclass(frozen=True)
class SWEPrediction:
    instance_id: str
    model_name_or_path: str
    model_patch: str

    def __post_init__(self) -> None:
        if not self.instance_id or not self.model_name_or_path:
            raise SWEGymAdapterError("prediction identifiers must not be empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "instance_id": self.instance_id,
            "model_name_or_path": self.model_name_or_path,
            "model_patch": self.model_patch,
        }


def load_swe_gym_jsonl(path: str | Path) -> tuple[SWEGymInstance, ...]:
    target = Path(path)
    instances: list[SWEGymInstance] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise SWEGymAdapterError("row must be a JSON object")
            instance = SWEGymInstance.from_dict(value)
        except (json.JSONDecodeError, SWEGymAdapterError) as exc:
            raise SWEGymAdapterError(f"line {line_number}: {exc}") from exc
        if instance.instance_id in seen:
            raise SWEGymAdapterError(f"line {line_number}: duplicate instance_id {instance.instance_id}")
        seen.add(instance.instance_id)
        instances.append(instance)
    if not instances:
        raise SWEGymAdapterError("SWE-Gym dataset must not be empty")
    return tuple(instances)


def write_predictions(predictions: Iterable[SWEPrediction], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps(prediction.to_dict(), ensure_ascii=False) for prediction in predictions]
    target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def build_swebench_harness_command(
    *,
    dataset_name: str,
    predictions_path: str | Path,
    run_id: str,
    instance_ids: tuple[str, ...] = (),
    max_workers: int = 1,
    cache_level: str = "env",
    clean: bool = True,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    if not dataset_name.strip():
        raise SWEGymAdapterError("dataset_name must not be empty")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise SWEGymAdapterError("run_id may contain only letters, numbers, dot, underscore and hyphen")
    if max_workers <= 0:
        raise SWEGymAdapterError("max_workers must be positive")
    if cache_level not in {"none", "base", "env", "instance"}:
        raise SWEGymAdapterError("unsupported cache_level")
    command = [
        python_executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--predictions_path",
        str(Path(predictions_path)),
        "--max_workers",
        str(max_workers),
        "--cache_level",
        cache_level,
        "--run_id",
        run_id,
        "--clean",
        str(clean),
    ]
    if instance_ids:
        command.extend(("--instance_ids", *instance_ids))
    return tuple(command)


def _required_string(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise SWEGymAdapterError(f"{key} must be a non-empty string")
    return result


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SWEGymAdapterError("image_name must be a non-empty string when provided")
    return value


def _string_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = json.loads(value) if value.strip() else []
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise SWEGymAdapterError("test lists must be arrays of strings or JSON-encoded arrays")
    return tuple(value)
