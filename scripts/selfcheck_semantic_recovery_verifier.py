from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from coding_agent_rl_lab.contracts import AgentAction, DatasetSplit
from coding_agent_rl_lab.docker_environment import DockerSandboxConfig, DockerSandboxEnvironment
from coding_agent_rl_lab.providers import DockerSandboxProvider, SubprocessCommandRunner
from coding_agent_rl_lab.reward_shaping import build_training_reward
from coding_agent_rl_lab.swe_gym import SWEGymTaskAdapter, audited_swe_gym_test_command


class SelfcheckError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-check a verifier with an executed recovery")
    parser.add_argument("--rows-cache", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--replay-spec", required=True)
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SelfcheckError(f"refusing to overwrite existing output: {output}")

    row = next(
        (
            item
            for item in _jsonl(Path(args.rows_cache))
            if item.get("instance_id") == args.task_id
        ),
        None,
    )
    if row is None:
        raise SelfcheckError(f"task not found in rows cache: {args.task_id}")
    spec = json.loads(Path(args.replay_spec).read_text(encoding="utf-8"))
    state = next(
        (item for item in spec.get("states", []) if item.get("state_id") == args.state_id),
        None,
    )
    if state is None:
        raise SelfcheckError(f"state not found in replay spec: {args.state_id}")
    actions = state.get("teacher_actions")
    if not isinstance(actions, list) or not actions:
        raise SelfcheckError("selected state has no teacher actions")

    adapter = SWEGymTaskAdapter()
    bundle = adapter.adapt(
        row,
        split=DatasetSplit.DEVELOPMENT,
        test_command=audited_swe_gym_test_command(row),
    )
    config = DockerSandboxConfig(
        memory_limit="4g",
        cpu_limit=2.0,
        pids_limit=512,
        startup_timeout_seconds=180.0,
        command_timeout_seconds=120.0,
        test_timeout_seconds=300.0,
    )
    runner = SubprocessCommandRunner()
    provider = DockerSandboxProvider(
        {bundle.task.task_id: bundle.environment}, config=config, runner=runner
    )
    environment = provider.create(bundle.task)
    if not isinstance(environment, DockerSandboxEnvironment):
        raise SelfcheckError("provider did not create a DockerSandboxEnvironment")

    action_records: list[dict[str, Any]] = []
    final_result = None
    try:
        baseline_observation = environment.reset(bundle.task)
        head = environment._exec(("git", "rev-parse", "HEAD"))
        test_hashes: dict[str, str] = {}
        for path in sorted(environment._protected_files):
            result = environment._exec(("sha256sum", path))
            if not result.passed:
                raise SelfcheckError(f"failed to hash verifier file {path}")
            test_hashes[path] = result.stdout.split()[0]
        for sequence, raw_action in enumerate(actions):
            action = AgentAction.from_dict(raw_action)
            final_result = environment.step(action)
            action_records.append(
                {
                    "sequence": sequence,
                    "action": raw_action,
                    "observation": final_result.observation,
                    "terminated": final_result.terminated,
                }
            )
            if final_result.terminated and sequence != len(actions) - 1:
                raise SelfcheckError("verifier self-check terminated before the final action")
        if final_result is None or not final_result.terminated or final_result.test_result is None:
            raise SelfcheckError("verifier self-check did not terminate with a test result")
        reward = build_training_reward(
            baseline=environment.baseline_result,
            final=final_result.test_result,
            patch_created=bool(environment.changed_files()),
            patch_valid=environment.patch_is_valid(),
            verifier_run_after_patch=True,
            violations=tuple(environment.violations),
            reward_version="conservative-v2",
        )
        changed_files = sorted(environment.changed_files())
        changed_hashes: dict[str, str] = {}
        for path in changed_files:
            result = environment._exec(("sha256sum", path))
            if result.passed:
                changed_hashes[path] = result.stdout.split()[0]
        image = runner.run(("docker", "image", "inspect", "--format", "{{.Id}}", bundle.environment.image))
        if not image.passed:
            raise SelfcheckError("failed to inspect verifier Docker image")
        report = {
            "schema_version": 1,
            "task_id": bundle.task.task_id,
            "base_commit_expected": bundle.task.base_commit,
            "base_commit_observed": head.stdout.strip(),
            "docker_image": bundle.environment.image,
            "docker_image_id": image.stdout.strip(),
            "resource_limits": {
                "network": config.network,
                "memory": config.memory_limit,
                "cpus": config.cpu_limit,
                "pids": config.pids_limit,
                "test_timeout_seconds": config.test_timeout_seconds,
            },
            "test_command": list(bundle.task.test_command),
            "test_patch_sha256": hashlib.sha256(row["test_patch"].encode("utf-8")).hexdigest(),
            "protected_test_file_sha256": test_hashes,
            "fail_to_pass": list(bundle.environment.fail_to_pass),
            "pass_to_pass": list(bundle.environment.pass_to_pass),
            "baseline_observation": baseline_observation,
            "actions": action_records,
            "changed_source_file_sha256": changed_hashes,
            "reward_components": reward.to_dict(),
            "completed": reward.strict_success,
        }
    finally:
        environment.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "completed": report["completed"],
        "base_commit_observed": report["base_commit_observed"],
        "docker_image_id": report["docker_image_id"],
        "protected_test_file_sha256": report["protected_test_file_sha256"],
        "reward_components": report["reward_components"],
    }, ensure_ascii=False, indent=2))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
