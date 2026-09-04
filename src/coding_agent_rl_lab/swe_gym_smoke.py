from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from coding_agent_rl_lab.contracts import DatasetSplit
from coding_agent_rl_lab.docker_environment import (
    DockerSandboxConfig,
    DockerSandboxEnvironment,
    SubprocessCommandRunner,
)
from coding_agent_rl_lab.swe_gym import SWEGymTaskAdapter, audited_swe_gym_test_command


DATASET_ROWS_URL_TEMPLATE = (
    "https://datasets-server.huggingface.co/rows?dataset=SWE-Gym%2FSWE-Gym"
    "&config=default&split=train&offset={offset}&length=1"
)


@dataclass(frozen=True)
class PinnedSWEGymRow:
    offset: int
    instance_id: str
    base_commit: str


PINNED_DEVELOPMENT_ROWS = (
    PinnedSWEGymRow(0, "getmoto__moto-7365", "7f6c9cb1deafb280fe7fcc7551c38e397f11a706"),
    PinnedSWEGymRow(21, "getmoto__moto-7514", "f14749b6b5f072ae42d98939e3948e09cd0f6a20"),
    PinnedSWEGymRow(63, "getmoto__moto-7646", "98ad497eb4de25943ceadabe7c2c5c73cc810e27"),
    PinnedSWEGymRow(115, "getmoto__moto-7446", "b13e493823397d3c9aae3598b3b024307f6b0974"),
    PinnedSWEGymRow(134, "getmoto__moto-7607", "ca24f65f2ff415a7eb7eddcea3bd8173e5d5f920"),
    PinnedSWEGymRow(148, "getmoto__moto-7509", "b22683eb98cb3de13d44f5348665f44539afa36f"),
    PinnedSWEGymRow(172, "getmoto__moto-7385", "6505971ecb0dda833b9828ba71522f2805655674"),
    PinnedSWEGymRow(204, "getmoto__moto-7608", "62e44dbd336227c87ef0eca0bbcb64f29f203300"),
    PinnedSWEGymRow(225, "getmoto__moto-7393", "dbfa456dda04f995479c995a361b60f521e1753e"),
    PinnedSWEGymRow(297, "getmoto__moto-7537", "d767a2799bb1a6a6f2ed05ceb7cbb0c789968b76"),
)

SWE_GYM_TASK_SET_ROWS = {
    "all": PINNED_DEVELOPMENT_ROWS,
    # Curriculum order starts with the only pinned task where the baseline policy
    # has previously produced a source patch; membership remains the first six rows.
    "train": (PINNED_DEVELOPMENT_ROWS[5], *PINNED_DEVELOPMENT_ROWS[:5]),
    "regression": PINNED_DEVELOPMENT_ROWS[6:8],
    "held-out": PINNED_DEVELOPMENT_ROWS[8:],
}
SWE_GYM_TASK_SET_CHOICES = tuple(SWE_GYM_TASK_SET_ROWS)
PINNED_INSTANCE_IDS = tuple(item.instance_id for item in PINNED_DEVELOPMENT_ROWS)

INSTANCE_ID = PINNED_DEVELOPMENT_ROWS[0].instance_id
BASE_COMMIT = PINNED_DEVELOPMENT_ROWS[0].base_commit


def main() -> int:
    print(f"[1/5] Downloading official SWE-Gym row for {INSTANCE_ID}...", flush=True)
    row = download_pinned_row()
    test_command = audited_swe_gym_test_command(row)
    bundle = SWEGymTaskAdapter().adapt(
        row,
        split=DatasetSplit.DEVELOPMENT,
        test_command=test_command,
    )

    print(f"[2/5] Pulling {bundle.environment.image}...", flush=True)
    subprocess.run(("docker", "pull", bundle.environment.image), check=True)

    config = DockerSandboxConfig(
        memory_limit="4g",
        cpu_limit=2.0,
        pids_limit=512,
        startup_timeout_seconds=180.0,
        command_timeout_seconds=120.0,
        test_timeout_seconds=900.0,
    )
    runner = SubprocessCommandRunner()
    environment = DockerSandboxEnvironment(bundle.environment, config, runner)
    try:
        print("[3/5] Starting the restricted container and verifying fail-before...", flush=True)
        environment.reset(bundle.task)
        baseline = environment.baseline_result
        if baseline is None or baseline.passed:
            raise RuntimeError("expected the patched baseline tests to fail")
        print(f"      Baseline failed as expected (exit={baseline.exit_code}).", flush=True)

        print("[4/5] Applying the verifier-only gold patch and verifying pass-after...", flush=True)
        gold_patch = row.get("patch")
        if not isinstance(gold_patch, str) or not gold_patch.strip():
            raise RuntimeError("official row does not contain a gold patch")
        container_name = environment.container_name
        if container_name is None:
            raise RuntimeError("Docker sandbox is not active")
        applied = runner.run(
            config.exec_argv(
                container_name,
                bundle.environment.repository_path,
                ("git", "apply", "--whitespace=nowarn", "-"),
                interactive=True,
            ),
            input_text=gold_patch,
            timeout_seconds=config.command_timeout_seconds,
        )
        if not applied.passed:
            raise RuntimeError(f"gold patch failed to apply: {applied.stderr or applied.stdout}")
        final = environment.finalize()
        if not final.passed:
            detail = final.stderr or final.stdout
            raise RuntimeError(f"gold patch did not pass audited tests:\n{detail[-8000:]}")
        print(f"      Gold patch passed the audited tests (exit={final.exit_code}).", flush=True)
    finally:
        environment.close()

    print("[5/5] PASS: real SWE-Gym image lifecycle and cleanup completed.", flush=True)
    print("      Gold/reference patch was used only for infrastructure validation, not as a model result.")
    return 0


def download_pinned_row() -> dict[str, Any]:
    return _download_pinned_row(PINNED_DEVELOPMENT_ROWS[0])


def download_pinned_rows(*, limit: int | None = None) -> tuple[dict[str, Any], ...]:
    if limit is not None and (limit <= 0 or limit > len(PINNED_DEVELOPMENT_ROWS)):
        raise ValueError(f"limit must be between 1 and {len(PINNED_DEVELOPMENT_ROWS)}")
    selected = PINNED_DEVELOPMENT_ROWS[:limit]
    rows: list[dict[str, Any]] = []
    for pinned in selected:
        row = dict(_download_pinned_row(pinned))
        row.pop("patch", None)
        row.pop("hints_text", None)
        rows.append(row)
    return tuple(rows)


def pinned_rows_for_task_set(task_set: str) -> tuple[PinnedSWEGymRow, ...]:
    try:
        return SWE_GYM_TASK_SET_ROWS[task_set]
    except KeyError as exc:
        choices = ", ".join(SWE_GYM_TASK_SET_CHOICES)
        raise ValueError(f"unknown SWE-Gym task set {task_set!r}; choose from {choices}") from exc


def dataset_split_for_task_set(task_set: str) -> DatasetSplit:
    pinned_rows_for_task_set(task_set)
    if task_set == "regression":
        return DatasetSplit.REGRESSION
    if task_set == "held-out":
        return DatasetSplit.HELD_OUT
    return DatasetSplit.DEVELOPMENT


def select_pinned_rows(
    task_set: str,
    *,
    limit: int | None = None,
    task_ids: Sequence[str] = (),
) -> tuple[PinnedSWEGymRow, ...]:
    available = pinned_rows_for_task_set(task_set)
    if task_ids:
        if limit is not None:
            raise ValueError("limit and task_ids cannot be used together")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("task_ids must not contain duplicates")
        available_by_id = {item.instance_id: item for item in available}
        outside = [task_id for task_id in task_ids if task_id not in available_by_id]
        if outside:
            raise ValueError(
                f"task ids are outside task set {task_set}: {', '.join(outside)}"
            )
        return tuple(available_by_id[task_id] for task_id in task_ids)
    if limit is None:
        return available
    if limit <= 0 or limit > len(available):
        raise ValueError(f"limit must be between 1 and {len(available)} for task set {task_set}")
    return available[:limit]


def load_or_download_pinned_rows(
    path: str | Path,
    *,
    limit: int | None = None,
    task_set: str = "all",
    task_ids: Sequence[str] = (),
) -> tuple[dict[str, Any], ...]:
    selected = select_pinned_rows(task_set, limit=limit, task_ids=task_ids)
    target = Path(path)
    cached_by_id: dict[str, dict[str, Any]] = {}
    if target.exists():
        cached = tuple(
            json.loads(line)
            for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        pinned_by_id = {item.instance_id: item for item in PINNED_DEVELOPMENT_ROWS}
        for row in cached:
            instance_id = row.get("instance_id")
            if not isinstance(instance_id, str) or instance_id not in pinned_by_id:
                raise RuntimeError("cached rollout row is outside the pinned SWE-Gym task set")
            if instance_id in cached_by_id:
                raise RuntimeError(f"cached rollout row is duplicated: {instance_id}")
            _validate_pinned_row(row, pinned_by_id[instance_id])
            if "patch" in row or "hints_text" in row:
                raise RuntimeError("cached rollout rows must not contain gold patch or hints")
            cached_by_id[instance_id] = row
        if all(item.instance_id in cached_by_id for item in selected):
            return tuple(cached_by_id[item.instance_id] for item in selected)

    for pinned in selected:
        if pinned.instance_id in cached_by_id:
            continue
        row = dict(_download_pinned_row(pinned))
        row.pop("patch", None)
        row.pop("hints_text", None)
        cached_by_id[pinned.instance_id] = row

    rows = tuple(
        cached_by_id[item.instance_id]
        for item in PINNED_DEVELOPMENT_ROWS
        if item.instance_id in cached_by_id
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return tuple(cached_by_id[item.instance_id] for item in selected)


def _download_pinned_row(pinned: PinnedSWEGymRow, *, attempts: int = 3) -> dict[str, Any]:
    last_error: OSError | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            DATASET_ROWS_URL_TEMPLATE.format(offset=pinned.offset),
            headers={"User-Agent": "coding-agent-rl-lab/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
            break
        except OSError as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    else:
        raise RuntimeError(
            f"failed to download pinned SWE-Gym row {pinned.instance_id} after {attempts} attempts"
        ) from last_error
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        raise RuntimeError("Hugging Face rows API returned an unexpected response")
    row = rows[0].get("row")
    if not isinstance(row, dict):
        raise RuntimeError("Hugging Face response does not contain a dataset row")
    _validate_pinned_row(row, pinned)
    return row


def _validate_pinned_row(row: dict[str, Any], pinned: PinnedSWEGymRow) -> None:
    if row.get("instance_id") != pinned.instance_id or row.get("base_commit") != pinned.base_commit:
        raise RuntimeError("downloaded row does not match the pinned smoke instance")
    if row.get("repo") != "getmoto/moto" or row.get("version") != "5.0":
        raise RuntimeError("downloaded row is outside the audited getmoto/moto@5.0 environment")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"SMOKE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
