"""Verify fail-before and reference-patch pass-after for pinned SWE-Gym rows.

`coding_agent_rl_lab.swe_gym_smoke` does this for one hard-coded task (`getmoto__moto-7365`).
A diagnostic that claims a task has a trustworthy `fail-before` needs the same evidence for every
task it uses, in the same environment, at the same code revision - so this script parameterises it.

Per task it starts the restricted container, applies the task's official test patch (which
`DockerSandboxEnvironment.reset` does and refuses if the baseline already passes), records the
baseline result, applies the **gold patch outside the model's reach**, and runs the audited test
command again. Nothing here is a model result: the gold patch exists only to prove the environment
can distinguish a broken repository from a repaired one.

The gold patch is answer-bearing and is read from a file the caller names - normally
`work/private/pinned-gold-patches.json`, which is gitignored. It never enters a task, a prompt or a
trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coding_agent_rl_lab.contracts import DatasetSplit  # noqa: E402
from coding_agent_rl_lab.docker_environment import (  # noqa: E402
    DockerSandboxConfig,
    DockerSandboxEnvironment,
    SubprocessCommandRunner,
)
from coding_agent_rl_lab.swe_gym import (  # noqa: E402
    SWEGymAdapterConfig,
    SWEGymTaskAdapter,
    audited_swe_gym_test_command,
)
from coding_agent_rl_lab.swe_gym_smoke import load_or_download_pinned_rows  # noqa: E402

VERDICT_VERIFIED = "verified"
VERDICT_BASELINE_PASSED = "failed_baseline_already_passes"
VERDICT_APPLY_FAILED = "failed_gold_patch_did_not_apply"
VERDICT_GOLD_FAILED = "failed_gold_patch_did_not_pass"
VERDICT_INFRASTRUCTURE = "infrastructure_error"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tail(value: str, limit: int = 4000) -> str:
    return value[-limit:]


def _image_id(docker_binary: str, image: str) -> str | None:
    try:
        completed = subprocess.run(
            (docker_binary, "image", "inspect", "--format", "{{.Id}}", image),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def check_task(
    row: dict[str, Any],
    gold_patch: str,
    *,
    test_timeout_seconds: float,
    pull: bool,
    max_steps: int,
) -> dict[str, Any]:
    task_id = row["instance_id"]
    adapter = SWEGymTaskAdapter(SWEGymAdapterConfig(max_steps=max_steps))
    bundle = adapter.adapt(
        row,
        split=DatasetSplit.DEVELOPMENT,
        test_command=audited_swe_gym_test_command(row),
    )
    spec = bundle.environment
    config = DockerSandboxConfig(
        memory_limit="4g",
        cpu_limit=2.0,
        pids_limit=512,
        startup_timeout_seconds=180.0,
        command_timeout_seconds=120.0,
        test_timeout_seconds=test_timeout_seconds,
    )
    runner = SubprocessCommandRunner()
    record: dict[str, Any] = {
        "task_id": task_id,
        "base_commit": spec.base_commit,
        "image": spec.image,
        "image_id": _image_id(config.docker_binary, spec.image),
        "test_command": list(bundle.task.test_command),
        "fail_to_pass": list(spec.fail_to_pass),
        "pass_to_pass": list(spec.pass_to_pass),
        "gold_patch_sha256": hashlib.sha256(gold_patch.encode("utf-8")).hexdigest(),
        "gold_patch_lines": len(gold_patch.splitlines()),
        "verdict": VERDICT_INFRASTRUCTURE,
        "failure": None,
    }
    if pull:
        record["pull"] = subprocess.run(
            (config.docker_binary, "pull", spec.image), capture_output=True, text=True
        ).returncode
    environment = DockerSandboxEnvironment(spec, config, runner)
    started = time.monotonic()
    try:
        environment.reset(bundle.task)
        baseline = environment.baseline_result
        record["baseline_passed"] = None if baseline is None else baseline.passed
        record["baseline_exit_code"] = None if baseline is None else baseline.exit_code
        record["baseline_duration_ms"] = None if baseline is None else baseline.duration_ms
        record["baseline_output_tail"] = "" if baseline is None else _tail(
            baseline.stderr or baseline.stdout
        )
        if baseline is None:
            record["verdict"] = VERDICT_INFRASTRUCTURE
            record["failure"] = "reset produced no baseline result"
            return record
        if baseline.passed:
            record["verdict"] = VERDICT_BASELINE_PASSED
            record["failure"] = "the baseline already passes on this image and test patch"
            return record
        container_name = environment.container_name
        if container_name is None:
            record["failure"] = "no container is active"
            return record
        applied = runner.run(
            config.exec_argv(
                container_name,
                spec.repository_path,
                ("git", "apply", "--whitespace=nowarn", "-"),
                interactive=True,
            ),
            input_text=gold_patch,
            timeout_seconds=config.command_timeout_seconds,
        )
        record["gold_applied"] = applied.passed
        record["gold_apply_output_tail"] = _tail(applied.stderr or applied.stdout)
        if not applied.passed:
            record["verdict"] = VERDICT_APPLY_FAILED
            record["failure"] = "the gold patch did not apply to the base commit"
            return record
        final = environment.finalize()
        record["final_passed"] = final.passed
        record["final_exit_code"] = final.exit_code
        record["final_duration_ms"] = final.duration_ms
        record["final_output_tail"] = _tail(final.stderr or final.stdout)
        record["verdict"] = VERDICT_VERIFIED if final.passed else VERDICT_GOLD_FAILED
        if not final.passed:
            record["failure"] = "the gold patch did not pass the audited test command"
        return record
    except Exception as exc:  # infrastructure is recorded, never hidden
        record["verdict"] = VERDICT_INFRASTRUCTURE
        record["failure"] = f"{type(exc).__name__}: {str(exc)[:2000]}"
        return record
    finally:
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
        environment.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-cache", default="work/swe-gym-development-rows.jsonl")
    parser.add_argument("--gold-patches", required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--pull", action="store_true", help="docker pull each image first")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows_cache = Path(args.rows_cache)
    gold_path = Path(args.gold_patches)
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    rows = load_or_download_pinned_rows(
        rows_cache, limit=None, task_set="all", task_ids=args.task_id
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pinned-fail-before-and-gold-pass-after",
        "rows_cache": str(rows_cache),
        "gold_patches": str(gold_path),
        "gold_patches_sha256": _sha256(gold_path),
        "test_timeout_seconds": args.test_timeout_seconds,
        "tasks": [],
        "run_complete": False,
    }
    for row in rows:
        task_id = row["instance_id"]
        if task_id not in gold:
            raise SystemExit(f"no gold patch for {task_id} in {gold_path}")
        print(f"[check] {task_id}", flush=True)
        record = check_task(
            row,
            gold[task_id],
            test_timeout_seconds=args.test_timeout_seconds,
            pull=args.pull,
            max_steps=args.max_steps,
        )
        report["tasks"].append(record)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"    verdict={record['verdict']} baseline_passed={record.get('baseline_passed')} "
            f"final_passed={record.get('final_passed')} "
            f"elapsed={record.get('elapsed_seconds')}s failure={record.get('failure')}",
            flush=True,
        )
    report["run_complete"] = True
    report["verified_count"] = sum(
        record["verdict"] == VERDICT_VERIFIED for record in report["tasks"]
    )
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"verified": report["verified_count"], "of": len(report["tasks"])}))
    return 0 if report["verified_count"] == len(report["tasks"]) else 1


if __name__ == "__main__":
    sys.exit(main())
