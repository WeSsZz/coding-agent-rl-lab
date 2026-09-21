"""Prove the reference repair is still reachable through the agent's own tools.

`check_pinned_gold.py` applies the gold patch with `git apply`, which only shows that the repository
and the verifier work. It says nothing about whether the *edit protocol* can express the repair. Once
the protocol gained a rule - `replace_lines` may only rewrite lines the policy has actually been
shown - that had to be re-checked, because a rule that makes the reference repair unreachable would
turn every arm into a zero for reasons that have nothing to do with the model.

So this drives the patch through `read_file` -> `replace_lines` -> `run_tests` inside the restricted
container, one hunk at a time, and requires the audited test command to pass at the end. It is an
infrastructure self-check: the patch is the answer, so nothing here is a model result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coding_agent_rl_lab.contracts import ActionKind, AgentAction, DatasetSplit  # noqa: E402
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

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")
SPAN_HINT = re.compile(r"spans lines (\d+)-(\d+)")
READ_CONTEXT_BEFORE = 10
READ_MINIMUM = 20
NUMBERED = re.compile(r"^(\d+): (.*)$", re.MULTILINE)


def parse_span_hint(observation: str) -> tuple[int, int] | None:
    """The enclosing statement the syntax gate names, e.g. "spans lines 371-434"."""

    match = SPAN_HINT.search(observation)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def numbered_lines(observation: str) -> dict[int, str]:
    return {int(number): text for number, text in NUMBERED.findall(observation)}


def replacement_for(
    shown: dict[int, str],
    hunks: Sequence[dict[str, Any]],
) -> str:
    """The new text for a span: the lines as read, with every contained hunk applied in place."""

    by_start = {hunk["old_start"]: hunk for hunk in hunks}
    ordered = sorted(shown)
    out: list[str] = []
    index = 0
    while index < len(ordered):
        number = ordered[index]
        hunk = by_start.get(number)
        if hunk is None:
            out.append(shown[number])
            index += 1
            continue
        out.extend(hunk["new_lines"])
        index += hunk["old_count"]
    return "\n".join(out)


def patch_hunks_with_bodies(patch: str) -> dict[str, list[dict[str, Any]]]:
    """Per file, each hunk as an old-side range plus the new-side text that replaces it.

    The replacement is the hunk's context and added lines in order - not just the added lines.
    `replace_lines` swaps the whole old range, so dropping context would delete working code.
    """

    hunks: dict[str, list[dict[str, Any]]] = {}
    current: str | None = None
    body: list[str] | None = None
    for line in patch.splitlines():
        if line.startswith("+++ "):
            value = line[4:].strip().split("\t", 1)[0]
            current = value[2:] if value.startswith("b/") else value
            if current in {"/dev/null", "dev/null"}:
                current = None
            body = None
            continue
        match = HUNK.match(line)
        if match and current is not None:
            old_start = int(match.group(1))
            old_count = int(match.group(2)) if match.group(2) is not None else 1
            body = []
            hunks.setdefault(current, []).append(
                {
                    "old_start": old_start,
                    "old_count": old_count,
                    "old_lines": [],
                    "new_lines": body,
                }
            )
            continue
        if body is None:
            continue
        if line.startswith(" "):
            body.append(line[1:])
            hunks[current][-1]["old_lines"].append(line[1:])
        elif line.startswith("+"):
            body.append(line[1:])
        elif line.startswith("-"):
            hunks[current][-1]["old_lines"].append(line[1:])
        elif line.startswith("\\"):
            continue
    return hunks


def read_window(hunk: dict[str, Any]) -> tuple[int, int]:
    """A window that contains the hunk and satisfies the tool's 20-line minimum."""

    old_start, old_count = hunk["old_start"], hunk["old_count"]
    old_end = old_start + max(old_count, 1) - 1
    start = max(1, old_start - READ_CONTEXT_BEFORE)
    end = max(start + READ_MINIMUM - 1, old_end + READ_CONTEXT_BEFORE)
    return start, end


def read_span(
    environment: Any,
    path: str,
    start: int,
    end: int,
) -> tuple[dict[int, str], str]:
    """Read exactly the span, padded only as far as the tool's 20-line minimum requires."""

    low, high = start, end
    while high - low + 1 < READ_MINIMUM:
        if low > 1:
            low -= 1
        else:
            high += 1
    read = environment.step(
        AgentAction(
            ActionKind.READ_FILE,
            {"path": path, "start_line": low, "end_line": high},
        )
    )
    return numbered_lines(read.observation), read.observation


def apply_text_fallback(
    environment: Any,
    path: str,
    start: int,
    end: int,
    hunks: Sequence[dict[str, Any]],
) -> tuple[Any, bool, str]:
    """`replace_text` with the statement's exact text, for a statement `replace_lines` cannot take.

    A statement longer than the 80-line `replace_lines` cap is still reachable: `replace_text` has no
    range limit, it just needs the exact old text and a result that parses. Whether the policy can
    actually *emit* that much text is a separate question - this records how big it had to be.
    """

    shown, observation = read_span(environment, path, start, end)
    missing = [number for number in range(start, end + 1) if number not in shown]
    if missing:
        return None, False, (
            f"the {end - start + 1}-line statement cannot be obtained in one read: "
            f"{len(missing)} of its lines were clipped by the read budget "
            f"(first missing {missing[0]})"
        )
    old = "\n".join(shown[number] for number in range(start, end + 1))
    new = replacement_for({n: shown[n] for n in range(start, end + 1)}, hunks)
    edit = environment.step(
        AgentAction(ActionKind.REPLACE_TEXT, {"path": path, "old": old, "new": new})
    )
    return edit, edit.observation.startswith("Updated "), edit.observation[:400]


def check_task(
    row: dict[str, Any],
    patch: str,
    *,
    test_timeout_seconds: float,
    max_steps: int,
) -> dict[str, Any]:
    task_id = row["instance_id"]
    adapter = SWEGymTaskAdapter(SWEGymAdapterConfig(max_steps=max_steps))
    bundle = adapter.adapt(
        row, split=DatasetSplit.DEVELOPMENT, test_command=audited_swe_gym_test_command(row)
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
    record: dict[str, Any] = {
        "task_id": task_id,
        "base_commit": spec.base_commit,
        "gold_patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest(),
        "hunks": [],
        "verdict": "infrastructure_error",
        "failure": None,
    }
    environment = DockerSandboxEnvironment(spec, config, SubprocessCommandRunner())
    try:
        environment.reset(bundle.task)
        baseline = environment.baseline_result
        record["baseline_passed"] = None if baseline is None else baseline.passed
        if baseline is None or baseline.passed:
            record["failure"] = "baseline did not fail"
            return record
        for path, hunks in patch_hunks_with_bodies(patch).items():
            # Bottom-up within a file. Applying a hunk changes the line count above the next one, so
            # top-down edits aim `replace_lines` at lines that no longer hold what the patch says -
            # the same stale-line-number trap a policy falls into. `git apply` handles it by
            # tracking offsets; a policy that edits from the end of the file never has to.
            ordered = sorted(hunks, key=lambda item: item["old_start"], reverse=True)
            for hunk in ordered:
                if hunk["old_count"] == 0:
                    record["hunks"].append(
                        {
                            "path": path,
                            "old_start": hunk["old_start"],
                            "tool": "replace_text",
                            "note": "pure insertion: the hunk's own text is the only handle",
                        }
                    )
                old_text = "\n".join(hunk["old_lines"])
                new_text = "\n".join(hunk["new_lines"])
                edit = environment.step(
                    AgentAction(
                        ActionKind.REPLACE_TEXT,
                        {"path": path, "old": old_text, "new": new_text},
                    )
                )
                applied = edit.observation.startswith("Updated ")
                record["hunks"].append(
                    {
                        "path": path,
                        "old_start": hunk["old_start"],
                        "old_count": hunk["old_count"],
                        "tool": "replace_text",
                        "old_text_lines": len(hunk["old_lines"]),
                        "applied": applied,
                        "observation": edit.observation[:400],
                    }
                )
                if applied:
                    continue
                start = hunk["old_start"]
                end = start + max(hunk["old_count"], 1) - 1
                shown, _ = read_span(environment, path, start, end)
                missing = [n for n in range(start, end + 1) if n not in shown]
                if missing:
                    record["verdict"] = "span_not_readable_in_one_call"
                    record["failure"] = (
                        f"{path}: lines {start}-{end} were clipped by the read budget "
                        f"({len(missing)} missing, first {missing[0]})"
                    )
                    return record
                ranged = environment.step(
                    AgentAction(
                        ActionKind.REPLACE_LINES,
                        {
                            "path": path,
                            "start_line": start,
                            "end_line": end,
                            "new": "\n".join(hunk["new_lines"]),
                        },
                    )
                )
                ranged_applied = ranged.observation.startswith("Updated ")
                record["hunks"].append(
                    {
                        "path": path,
                        "old_start": start,
                        "old_end": end,
                        "tool": "replace_lines",
                        "applied": ranged_applied,
                        "observation": ranged.observation[:400],
                    }
                )
                if ranged_applied:
                    continue
                hint = parse_span_hint(ranged.observation)
                record["verdict"] = "edit_refused"
                record["failure"] = (
                    f"{path} {start}-{end}: {ranged.observation[:300]}"
                    + ("" if hint is None else f" (enclosing statement {hint[0]}-{hint[1]})")
                )
                return record
        result = environment.step(AgentAction(ActionKind.RUN_TESTS))
        test_result = result.test_result
        record["final_passed"] = None if test_result is None else test_result.passed
        record["final_exit_code"] = None if test_result is None else test_result.exit_code
        record["changed_files"] = list(environment.changed_files())
        record["violations"] = list(environment.violations)
        record["verdict"] = "reachable" if (test_result and test_result.passed) else "tests_still_fail"
        if not (test_result and test_result.passed):
            detail = "" if test_result is None else (test_result.stderr or test_result.stdout)
            record["failure"] = detail[-2000:]
        return record
    except Exception as exc:  # infrastructure is recorded, never hidden
        record["failure"] = f"{type(exc).__name__}: {str(exc)[:2000]}"
        return record
    finally:
        environment.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-cache", default="work/swe-gym-development-rows.jsonl")
    parser.add_argument("--gold-patches", required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-steps", type=int, default=400)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gold = json.loads(Path(args.gold_patches).read_text(encoding="utf-8"))
    rows = load_or_download_pinned_rows(
        Path(args.rows_cache), limit=None, task_set="all", task_ids=args.task_id
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "reference-repair-through-the-agent-tools",
        "disclosure": (
            "the gold patch is the answer; this is an infrastructure self-check that the edit "
            "protocol can express the repair, not a model result"
        ),
        "gold_patches_sha256": hashlib.sha256(Path(args.gold_patches).read_bytes()).hexdigest(),
        "tasks": [],
    }
    for row in rows:
        task_id = row["instance_id"]
        if task_id not in gold:
            raise SystemExit(f"no gold patch for {task_id}")
        print(f"[check] {task_id}", flush=True)
        record = check_task(
            row,
            gold[task_id],
            test_timeout_seconds=args.test_timeout_seconds,
            max_steps=args.max_steps,
        )
        report["tasks"].append(record)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"    verdict={record['verdict']} failure={record.get('failure')}", flush=True)
    report["reachable_count"] = sum(
        record["verdict"] == "reachable" for record in report["tasks"]
    )
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"reachable": report["reachable_count"], "of": len(report["tasks"])}))
    return 0 if report["reachable_count"] == len(report["tasks"]) else 1


if __name__ == "__main__":
    sys.exit(main())
