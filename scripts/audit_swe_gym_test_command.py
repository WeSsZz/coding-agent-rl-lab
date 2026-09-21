"""Audit a candidate pytest invocation for a SWE-Gym repository on the Docker worker.

The official SWE-Gym rows carry no test command, so every `(repo, version)` pair needs an
audited one before its tasks can run. This script validates a candidate *through the real
environment path* - the same `SWEGymTaskAdapter` and `DockerSandboxProvider` a rollout uses -
rather than through a bespoke probe. If the audit passes, the command is known to work in the
code that will actually execute it.

What "passes" means, per task:

* the container starts and `git rev-parse HEAD` equals the row's `base_commit`;
* the official test patch applies;
* the candidate command runs and exits non-zero with the baseline (fail-before);
* the set of failing node ids equals the row's `FAIL_TO_PASS`.

It does NOT apply the gold patch. Fail-before plus an exact failing-node match is what the
command needs to prove; pass-after is a separate self-check that already exists in
`scripts/check_pinned_gold.py` and needs a real edit path.

Run it on the Ubuntu worker (it needs Docker):

    PYTHONPATH=src python3 scripts/audit_swe_gym_test_command.py \
        --rows work/private/audit-rows.jsonl \
        --prefix /opt/miniconda3/envs/testbed/bin/pytest -n0 -rA \
        --output work/private/test-command-audit.json
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from coding_agent_rl_lab.contracts import DatasetSplit  # noqa: E402
from coding_agent_rl_lab.docker_environment import (  # noqa: E402
    DockerSandboxConfig,
    DockerSandboxProvider,
)
from coding_agent_rl_lab.environment import failing_nodes, verifier_output_streams  # noqa: E402
from coding_agent_rl_lab.swe_gym import SWEGymTaskAdapter  # noqa: E402

#: The audit is an infrastructure check, so the task's split label is irrelevant to it. It uses
#: DEVELOPMENT because that is the one split with no external meaning: `DatasetSplit` predates
#: the route-v2 train / development / final-test vocabulary and has no `TRAIN` member.
_AUDIT_SPLIT = DatasetSplit.DEVELOPMENT


def build_command(prefix: tuple[str, ...], fail_to_pass: tuple[str, ...], pass_to_pass: tuple[str, ...]) -> tuple[str, ...]:
    targets = tuple(dict.fromkeys((*fail_to_pass, *pass_to_pass)))
    return (*prefix, "--", *targets)


def _tail(text: str, limit: int = 2000) -> str:
    return text if len(text) <= limit else "..." + text[-limit:]


def audit_one(row: dict[str, Any], prefix: tuple[str, ...], timeout: int) -> dict[str, Any]:
    adapter = SWEGymTaskAdapter()
    fail_to_pass = tuple(row["FAIL_TO_PASS"])
    pass_to_pass = tuple(row["PASS_TO_PASS"])
    command = build_command(prefix, fail_to_pass, pass_to_pass)
    record: dict[str, Any] = {
        "instance_id": row["instance_id"],
        "repo": row["repo"],
        "version": row["version"],
        "base_commit": row["base_commit"],
        "image": adapter.image_for_instance(row["instance_id"]),
        "command": list(command),
        "fail_to_pass": list(fail_to_pass),
        "pass_to_pass_count": len(pass_to_pass),
    }
    bundle = adapter.adapt(row, split=_AUDIT_SPLIT, test_command=command)
    provider = DockerSandboxProvider(
        {bundle.task.task_id: bundle.environment},
        DockerSandboxConfig(test_timeout_seconds=timeout),
    )
    environment = None
    try:
        environment = provider.create(bundle.task)
        environment.reset(bundle.task)
        baseline = environment.baseline_result
        baseline_nodes = tuple(sorted(failing_nodes(verifier_output_streams(baseline))))
        record["baseline_exit_code"] = baseline.exit_code
        record["baseline_passed"] = baseline.passed
        record["baseline_failed_nodes"] = list(baseline_nodes)
        record["failing_set_matches_fail_to_pass"] = set(baseline_nodes) == set(fail_to_pass)
        record["missing_from_baseline"] = sorted(set(fail_to_pass) - set(baseline_nodes))
        record["unexpected_failures"] = sorted(set(baseline_nodes) - set(fail_to_pass))
        record["fail_before"] = not baseline.passed
        record["verified"] = bool(
            (not baseline.passed)
            and set(baseline_nodes) == set(fail_to_pass)
        )
        if not record["verified"]:
            # A non-zero exit with no `FAILED <node>` line means pytest never got as far as
            # reporting those nodes - a collection error, a usage error, or missing paths.
            # Without the raw output that is indistinguishable from "the test passed", so the
            # tool keeps a bounded tail of both streams whenever it cannot verify.
            record["baseline_stdout_tail"] = _tail(baseline.stdout)
            record["baseline_stderr_tail"] = _tail(baseline.stderr)
    except Exception as exc:  # the audit must record every failure mode, not abort the sweep
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["verified"] = False
    finally:
        if environment is not None:
            try:
                environment.close()
            except Exception as exc:  # pragma: no cover - cleanup reporting only
                record["close_error"] = f"{type(exc).__name__}: {exc}"
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument(
        "--prefix",
        action="append",
        required=True,
        dest="prefixes",
        metavar="[repo/version=]COMMAND",
        help=(
            "a candidate pytest invocation without the test nodes, as one shell-quoted string, "
            "e.g. '/opt/miniconda3/envs/testbed/bin/pytest -n0 -rA'. Scope it to one repository "
            "with 'repo/version=...'; an unscoped entry is the default for rows nothing scoped "
            "matches. Repeatable. Per-repository scoping is necessary, not cosmetic: images "
            "differ in which plugins are installed, so '-n0' (an xdist flag) makes pytest exit 4 "
            "on an image without xdist. It is one string because a pytest prefix contains "
            "options argparse would otherwise try to parse."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--only",
        action="append",
        dest="only",
        help="restrict the audit to these instance ids (repeatable)",
    )
    return parser


def parse_prefixes(
    entries: list[str],
) -> tuple[dict[tuple[str, str], tuple[str, ...]], tuple[str, ...] | None]:
    scoped: dict[tuple[str, str], tuple[str, ...]] = {}
    default: tuple[str, ...] | None = None
    for entry in entries:
        key, separator, value = entry.partition("=")
        if separator and "/" in key:
            # Repository names contain a slash (`conan-io/conan`), so scope on the LAST one.
            repo, _, version = key.rpartition("/")
            if not repo or not version:
                raise ValueError(f"malformed prefix scope {key!r}; expected repo/version")
            scoped[(repo, version)] = tuple(shlex.split(value))
            continue
        if default is not None:
            raise ValueError("at most one unscoped --prefix is allowed")
        default = tuple(shlex.split(entry))
    if default is not None and not default:
        raise ValueError("an unscoped --prefix must not be empty")
    if not scoped and default is None:
        raise ValueError("at least one --prefix is required")
    return scoped, default


def prefix_for(
    row: dict[str, Any],
    scoped: dict[tuple[str, str], tuple[str, ...]],
    default: tuple[str, ...] | None,
) -> tuple[str, ...]:
    found = scoped.get((row["repo"], row["version"]))
    if found:
        return found
    if default is None:
        raise ValueError(f"no prefix for {row['repo']}@{row['version']} and no default was given")
    return default


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    rows = [json.loads(line) for line in arguments.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    if arguments.only:
        wanted = set(arguments.only)
        rows = [row for row in rows if row["instance_id"] in wanted]
    if not rows:
        print("FAILED  no rows to audit")
        return 1

    scoped, default = parse_prefixes(arguments.prefixes)

    records = []
    for row in rows:
        prefix = prefix_for(row, scoped, default)
        print(f"[audit] {row['instance_id']} ({row['repo']}@{row['version']})")
        print(f"        prefix: {' '.join(prefix)}")
        record = audit_one(row, prefix, arguments.test_timeout_seconds)
        records.append(record)
        status = "verified" if record["verified"] else "NOT verified"
        print(f"        {status}: {record.get('error') or record.get('baseline_exit_code')}")
        if record.get("missing_from_baseline"):
            print(f"        never failed: {record['missing_from_baseline'][:3]}")
        if record.get("unexpected_failures"):
            print(f"        also failed: {record['unexpected_failures'][:3]}")

    report = {
        "schema_version": 1,
        "prefixes": {f"{repo}/{version}": list(value) for (repo, version), value in sorted(scoped.items())},
        "default_prefix": list(default) if default else None,
        "audited_at": "2026-09-22",
        "verified": sum(1 for record in records if record["verified"]),
        "total": len(records),
        "records": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{report['verified']}/{report['total']} verified; wrote {arguments.output}")
    return 0 if report["verified"] == report["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
