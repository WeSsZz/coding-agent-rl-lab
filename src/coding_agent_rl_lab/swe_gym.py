from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import CodingTask, DatasetSplit
from .docker_environment import DockerTaskSpec


class SWEGymAdapterError(ValueError):
    pass


SWE_GYM_ENVIRONMENT_REVISION = "242429c188fcfd06aad13fce9a54d450470bf0ac"

_AUDITED_TEST_COMMAND_PREFIXES: dict[tuple[str, str], tuple[str, ...]] = {
    ("getmoto/moto", "5.0"): (
        "/opt/miniconda3/envs/testbed/bin/pytest",
        "-n0",
        "-rA",
    ),
}


@dataclass(frozen=True)
class SWEGymTaskBundle:
    task: CodingTask
    environment: DockerTaskSpec


@dataclass(frozen=True)
class SWEGymAdapterConfig:
    dataset_id: str = "SWE-Gym/SWE-Gym"
    dataset_revision: str = "bb94ed9"
    image_namespace: str = "xingyaoww"
    image_prefix: str = "sweb.eval.x86_64"
    repository_path: str = "/testbed"
    max_steps: int = 40


class SWEGymTaskAdapter:
    """Maps official SWE-Gym rows without exposing gold or verifier patches to policies."""

    REQUIRED_FIELDS = (
        "instance_id",
        "problem_statement",
        "repo",
        "base_commit",
        "version",
        "test_patch",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
    )

    def __init__(self, config: SWEGymAdapterConfig | None = None) -> None:
        self.config = config or SWEGymAdapterConfig()

    def adapt(
        self,
        row: dict[str, Any],
        *,
        split: DatasetSplit,
        test_command: tuple[str, ...] | None = None,
    ) -> SWEGymTaskBundle:
        missing = [name for name in self.REQUIRED_FIELDS if name not in row or row[name] is None]
        if missing:
            raise SWEGymAdapterError(f"SWE-Gym row is missing required fields: {', '.join(missing)}")
        instance_id = self._required_string(row, "instance_id")
        issue = self._required_string(row, "problem_statement")
        repo = self._required_string(row, "repo")
        base_commit = self._required_string(row, "base_commit")
        version = self._required_string(row, "version")
        test_patch = self._required_string(row, "test_patch")
        fail_to_pass = self._string_list(row, "FAIL_TO_PASS", require_non_empty=True)
        pass_to_pass = self._string_list(row, "PASS_TO_PASS", require_non_empty=False)
        command = test_command or self._optional_command(row.get("test_command"))
        if not command:
            raise SWEGymAdapterError(
                "SWE-Gym rows do not contain a portable test command; provide a versioned, repo-specific command"
            )
        image = self.image_for_instance(instance_id)
        provenance = f"hf://datasets/{self.config.dataset_id}@{self.config.dataset_revision}"
        task = CodingTask(
            task_id=instance_id,
            issue=issue,
            fixture_path=None,
            base_commit=base_commit,
            test_command=command,
            split=split,
            provenance=provenance,
            max_steps=self.config.max_steps,
            metadata={
                "source": "swe_gym",
                "repo": repo,
                "version": version,
                "created_at": row.get("created_at"),
                "docker_image": image,
                "fail_to_pass_count": len(fail_to_pass),
                "pass_to_pass_count": len(pass_to_pass),
            },
        )
        environment = DockerTaskSpec(
            task_id=instance_id,
            image=image,
            base_commit=base_commit,
            test_patch=test_patch,
            repository_path=self.config.repository_path,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
        )
        return SWEGymTaskBundle(task=task, environment=environment)

    def image_for_instance(self, instance_id: str) -> str:
        if "__" not in instance_id:
            raise SWEGymAdapterError("instance_id must use the SWE-Gym owner__repo-number form")
        normalized = instance_id.replace("__", "_s_").lower()
        return f"{self.config.image_namespace}/{self.config.image_prefix}.{normalized}:latest"

    @staticmethod
    def _required_string(row: dict[str, Any], name: str) -> str:
        value = row.get(name)
        if not isinstance(value, str) or not value.strip():
            raise SWEGymAdapterError(f"{name} must be a non-empty string")
        return value

    @staticmethod
    def _string_list(row: dict[str, Any], name: str, *, require_non_empty: bool) -> tuple[str, ...]:
        value = row.get(name)
        if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
            raise SWEGymAdapterError(f"{name} must be a list of non-empty strings")
        if require_non_empty and not value:
            raise SWEGymAdapterError(f"{name} must not be empty")
        return tuple(value)

    @staticmethod
    def _optional_command(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
            raise SWEGymAdapterError("test_command must be a list of non-empty strings")
        return tuple(value)


def audited_swe_gym_test_command(row: dict[str, Any]) -> tuple[str, ...]:
    """Return a shell-free command pinned to audited SWE-Gym environment constants."""

    repo = SWEGymTaskAdapter._required_string(row, "repo")
    version = SWEGymTaskAdapter._required_string(row, "version")
    try:
        prefix = _AUDITED_TEST_COMMAND_PREFIXES[(repo, version)]
    except KeyError as exc:
        raise SWEGymAdapterError(f"no audited test command for {repo}@{version}") from exc
    fail_to_pass = SWEGymTaskAdapter._string_list(row, "FAIL_TO_PASS", require_non_empty=True)
    pass_to_pass = SWEGymTaskAdapter._string_list(row, "PASS_TO_PASS", require_non_empty=False)
    targets = tuple(dict.fromkeys((*fail_to_pass, *pass_to_pass)))
    if any(target.startswith("-") for target in targets):
        raise SWEGymAdapterError("test target must not be an option")
    return (*prefix, "--", *targets)


def load_swe_gym_jsonl(
    path: str | Path,
    *,
    split: DatasetSplit,
    adapter: SWEGymTaskAdapter | None = None,
) -> tuple[SWEGymTaskBundle, ...]:
    target = Path(path)
    selected = adapter or SWEGymTaskAdapter()
    bundles: list[SWEGymTaskBundle] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise SWEGymAdapterError("row must be a JSON object")
            bundle = selected.adapt(row, split=split)
        except (json.JSONDecodeError, SWEGymAdapterError) as exc:
            raise SWEGymAdapterError(f"invalid SWE-Gym row at line {line_number}: {exc}") from exc
        if bundle.task.task_id in seen:
            raise SWEGymAdapterError(f"duplicate SWE-Gym task at line {line_number}: {bundle.task.task_id}")
        seen.add(bundle.task.task_id)
        bundles.append(bundle)
    if not bundles:
        raise SWEGymAdapterError("SWE-Gym dataset must not be empty")
    return tuple(bundles)
