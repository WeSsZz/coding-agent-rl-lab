"""Summarise an A/B/C diagnostic run: arm totals plus the per-step behaviour behind them.

The point of the diagnostic is not the pass rate - one pass in six trials says little - but *where*
each condition stops. So this reports, per trial, both the grader's own numbers and the behaviour
that produced them:

* did the trial read an implementation file the reviewed repair touches?
* did it search a literal that its own initial observation quoted, or one only the repair contains?
* did it apply any edit at all, and did an applied edit land in a repair file?
* after the verifier first reported a failure, did a later edit target a *new* file, or repeat?
* how many `finish` refusals did it collect, and how many times did it run the verifier?

Nothing here is a capability claim. A condition that is assisted is labelled as assisted, and a
condition with two trials per task is a screen, not a measurement of a success rate.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

CONDITION_LABEL = {
    "A": "A autonomous",
    "B": "B oracle-file (assisted)",
    "C": "C oracle-context (assisted)",
}
EDIT_KINDS = {"replace_text", "replace_lines"}
VERIFIER_OBSERVATION_MARKERS = ("[failing tests]", "Tests failed", "Tests passed", "FAILED ")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _added_lines(patch: str, *, minimum_length: int = 12) -> list[str]:
    seen: list[str] = []
    for line in patch.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        candidate = line[1:].strip()
        if len(candidate) >= minimum_length and candidate not in seen:
            seen.append(candidate)
    return seen


def _refusal_reason(observation: str) -> str:
    head = observation.strip().splitlines()[0] if observation.strip() else ""
    head = re.sub(r"\s+", " ", head)
    return head[:160]


def refusal_class(reason: str) -> str:
    """Which kind of refusal this is, because they answer different questions.

    A read-before-edit refusal says the assist did not satisfy the environment's rule (the C window
    does not count as reading). An unparseable-edit refusal says the model found the right place and
    wrote a change that does not compile. A no-match refusal says it could not produce the exact text
    `replace_text` needs. Those are three different gaps behind one word, "refused".
    """

    if "requires reading the target file first" in reason:
        return "read_before_edit"
    if "unparseable" in reason:
        return "edit_would_not_parse"
    if "exactly one match" in reason:
        return "text_not_found_or_ambiguous"
    if "do not reread an unchanged file" in reason:
        return "repeated_unchanged_read"
    if "do not repeat the same action" in reason:
        return "repeated_refused_action"
    return "other"


def trial_indicators(
    trajectory: dict[str, Any],
    *,
    gold_paths: set[str],
    reference_literals: Sequence[str],
) -> dict[str, Any]:
    initial = trajectory.get("initial_observation") or ""
    steps = trajectory["steps"]
    read_paths: list[str] = []
    edits_applied: list[str] = []
    edits_refused: list[tuple[str, str]] = []
    queries: list[str] = []
    run_tests_actions = 0
    finish_refusals = 0
    first_failure_index: int | None = None
    retarget_after_failure = False
    tokens = 0
    actions_by_kind: collections.Counter = collections.Counter()
    refusals_by_class: collections.Counter = collections.Counter()
    refusal_streak = 0
    longest_refusal_streak = 0

    for index, step in enumerate(steps):
        action = step["action"]
        kind = action["kind"]
        arguments = action.get("arguments") or {}
        observation = step.get("observation") or ""
        usage = (step.get("policy_metadata") or {}).get("usage") or {}
        tokens += int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
        # A step carries `test_result` whenever one is known, so it is not a count of verifier
        # runs: every later `read_file` inherits the last result. Only a `run_tests` action runs it.
        if kind == "run_tests":
            run_tests_actions += 1
        actions_by_kind[kind] += 1
        if observation.startswith("Tool error"):
            refusal_streak += 1
            longest_refusal_streak = max(longest_refusal_streak, refusal_streak)
            refusals_by_class[refusal_class(observation)] += 1
        else:
            refusal_streak = 0

        if kind == "read_file":
            path = arguments.get("path")
            if isinstance(path, str) and path not in read_paths:
                read_paths.append(path)
        elif kind == "search_text":
            query = arguments.get("query")
            if isinstance(query, str):
                queries.append(query)
        elif kind in EDIT_KINDS:
            path = arguments.get("path")
            path = path if isinstance(path, str) else "<missing>"
            if observation.startswith("Updated "):
                if path not in edits_applied:
                    edits_applied.append(path)
            else:
                edits_refused.append((path, _refusal_reason(observation)))
        elif kind == "finish" and observation.startswith("Tool error"):
            finish_refusals += 1

        if kind == "run_tests" and step.get("test_result") is not None:
            passed = (step["test_result"] or {}).get("passed")
            if passed is False and first_failure_index is None:
                first_failure_index = index
        if (
            first_failure_index is not None
            and index > first_failure_index
            and kind in EDIT_KINDS
            and observation.startswith("Updated ")
        ):
            earlier = {path for path, _ in edits_refused}
            earlier.update(edits_applied[:-1] if edits_applied else [])
            if path not in earlier:
                retarget_after_failure = True

    lower_initial = initial.lower()
    searched_from_observation = [
        query for query in queries if len(query.strip()) >= 6 and query.strip().lower() in lower_initial
    ]
    searched_reference = [
        query
        for query in queries
        if any(literal in query or query in literal for literal in reference_literals)
    ]
    return {
        "read_paths": read_paths,
        "queries": queries,
        "edits_applied": edits_applied,
        "edits_refused": [
            {"path": path, "reason": reason} for path, reason in edits_refused
        ],
        "run_tests_actions": run_tests_actions,
        "actions_by_kind": dict(actions_by_kind),
        "refusals_by_class": dict(refusals_by_class),
        "refused_steps": sum(refusals_by_class.values()),
        "longest_refusal_streak": longest_refusal_streak,
        "finish_refusals": finish_refusals,
        "tokens": tokens,
        "read_a_repair_file": any(path in gold_paths for path in read_paths),
        "applied_an_edit": bool(edits_applied),
        "applied_an_edit_in_a_repair_file": any(path in gold_paths for path in edits_applied),
        "edit_refusals": len(edits_refused),
        "searched_a_literal_from_its_own_observation": searched_from_observation,
        "searched_reference_text": searched_reference,
        "retargeted_after_a_failing_verifier_run": retarget_after_failure,
    }


def build_rows(
    trajectories: Sequence[dict[str, Any]], gold: dict[str, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        task_id = trajectory["task_id"]
        paths = set(
            re.findall(r"^\+\+\+ b/(.+)$", gold.get(task_id, ""), re.M)
        )
        literals = _added_lines(gold.get(task_id, ""))
        verifier = trajectory.get("verifier") or {}
        indicators = trial_indicators(
            trajectory, gold_paths=paths, reference_literals=literals
        )
        rows.append(
            {
                "task_id": task_id,
                "condition": trajectory["policy"]["metadata"].get("diagnostic_condition"),
                "repetition": trajectory["repetition"],
                "seed": trajectory["seed"],
                "strict_success": bool(trajectory["reward"]["task_success"]),
                "tests_passed": bool(trajectory["reward"]["tests_passed"]),
                "changed_files": list(trajectory["changed_files"]),
                "steps": trajectory["reward"]["steps"],
                "loop_rejections": trajectory["reward"]["loop_rejections"],
                "violations": list(trajectory["reward"]["violations"]),
                "fail_to_pass_total": verifier.get("fail_to_pass_total"),
                "fail_to_pass_resolved": verifier.get("fail_to_pass_resolved"),
                "pass_to_pass_total": verifier.get("pass_to_pass_total"),
                "pass_to_pass_regressed": verifier.get("pass_to_pass_regressed"),
                "failed_nodes": verifier.get("failed_nodes") or [],
                "training_reward": trajectory.get("training_reward"),
                "auxiliary_chars": trajectory["policy"]["metadata"].get("auxiliary_chars"),
                **indicators,
            }
        )
    return rows


def _arm(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    trials = len(rows)
    return {
        "trial_count": trials,
        "strict_success_count": sum(row["strict_success"] for row in rows),
        "tests_passed_count": sum(row["tests_passed"] for row in rows),
        "trials_with_a_changed_file": sum(bool(row["changed_files"]) for row in rows),
        "trials_that_applied_an_edit": sum(row["applied_an_edit"] for row in rows),
        "trials_that_edited_a_repair_file": sum(
            row["applied_an_edit_in_a_repair_file"] for row in rows
        ),
        "trials_that_read_a_repair_file": sum(row["read_a_repair_file"] for row in rows),
        "fail_to_pass_resolved_total": sum(row["fail_to_pass_resolved"] or 0 for row in rows),
        "fail_to_pass_total": sum(row["fail_to_pass_total"] or 0 for row in rows),
        "mean_steps": round(statistics.fmean(row["steps"] for row in rows), 2) if rows else None,
        "mean_edit_refusals": round(
            statistics.fmean(row["edit_refusals"] for row in rows), 2
        )
        if rows
        else None,
        "mean_finish_refusals": round(
            statistics.fmean(row["finish_refusals"] for row in rows), 2
        )
        if rows
        else None,
        "mean_run_tests_actions": round(
            statistics.fmean(row["run_tests_actions"] for row in rows), 2
        )
        if rows
        else None,
        "trials_searching_own_observation_literal": sum(
            bool(row["searched_a_literal_from_its_own_observation"]) for row in rows
        ),
        "trials_searching_reference_text": sum(
            bool(row["searched_reference_text"]) for row in rows
        ),
        "trials_retargeting_after_a_failing_run": sum(
            row["retargeted_after_a_failing_verifier_run"] for row in rows
        ),
        "refused_steps": sum(row["refused_steps"] for row in rows),
        "mean_refused_steps": round(
            statistics.fmean(row["refused_steps"] for row in rows), 2
        )
        if rows
        else None,
        "longest_refusal_streak": max(
            (row["longest_refusal_streak"] for row in rows), default=0
        ),
        "refusals_by_class": dict(
            collections.Counter(
                name
                for row in rows
                for name, count in row["refusals_by_class"].items()
                for _ in range(count)
            )
        ),
        "actions_by_kind": dict(
            collections.Counter(
                name
                for row in rows
                for name, count in row["actions_by_kind"].items()
                for _ in range(count)
            )
        ),
        "tokens": sum(row["tokens"] for row in rows),
        "violations": sorted(
            {violation for row in rows for violation in row["violations"]}
        ),
        "mean_training_reward": round(
            statistics.fmean(
                row["training_reward"] for row in rows if row["training_reward"] is not None
            ),
            4,
        )
        if any(row["training_reward"] is not None for row in rows)
        else None,
    }


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    conditions = sorted({row["condition"] for row in rows})
    tasks = sorted({row["task_id"] for row in rows})
    by_task_condition = {
        task: {condition: _arm([r for r in rows if r["task_id"] == task and r["condition"] == condition])
               for condition in conditions}
        for task in tasks
    }
    return {
        "trials": len(rows),
        "by_condition": {condition: _arm([r for r in rows if r["condition"] == condition]) for condition in conditions},
        "by_task_condition": by_task_condition,
        "refusal_reasons": collections.Counter(
            entry["reason"] for row in rows for entry in row["edits_refused"]
        ).most_common(10),
        "queries": collections.Counter(
            query for row in rows for query in row["queries"]
        ).most_common(25),
    }


def markdown(rows: Sequence[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = ["# A/B/C diagnostic run", "", "## Per task and condition", ""]
    lines.append(
        "| task | condition | n | strict | tests | f2p | changed | applied | repair-file edit | "
        "read repair file | mean steps | edit refusals | finish refusals | run_tests actions |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for task, by_condition in summary["by_task_condition"].items():
        for condition, arm in sorted(by_condition.items()):
            lines.append(
                f"| `{task}` | {CONDITION_LABEL.get(condition, condition)} | {arm['trial_count']} | "
                f"{arm['strict_success_count']} | {arm['tests_passed_count']} | "
                f"{arm['fail_to_pass_resolved_total']}/{arm['fail_to_pass_total']} | "
                f"{arm['trials_with_a_changed_file']} | {arm['trials_that_applied_an_edit']} | "
                f"{arm['trials_that_edited_a_repair_file']} | {arm['trials_that_read_a_repair_file']} | "
                f"{arm['mean_steps']} | {arm['mean_edit_refusals']} | {arm['mean_finish_refusals']} | "
                f"{arm['mean_run_tests_actions']} |"
            )
    lines.extend(["", "## Arm totals (all three tasks)", ""])
    lines.append(
        "| condition | n | strict | tests | changed | applied edit | repair-file edit | "
        "searched own-observation literal | retargeted after failure | mean TR |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for condition, arm in sorted(summary["by_condition"].items()):
        lines.append(
            f"| {CONDITION_LABEL.get(condition, condition)} | {arm['trial_count']} | "
            f"{arm['strict_success_count']} | {arm['tests_passed_count']} | "
            f"{arm['trials_with_a_changed_file']} | {arm['trials_that_applied_an_edit']} | "
            f"{arm['trials_that_edited_a_repair_file']} | "
            f"{arm['trials_searching_own_observation_literal']} | "
            f"{arm['trials_retargeting_after_a_failing_run']} | {arm['mean_training_reward']} |"
        )
    lines.extend(["", "## Per trial", ""])
    lines.append(
        "| task | cond | rep | seed | steps | f2p | changed files | applied edits | "
        "refused | finish refusals | violations |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in sorted(rows, key=lambda r: (r["task_id"], r["condition"], r["repetition"])):
        lines.append(
            f"| `{row['task_id']}` | {row['condition']} | {row['repetition']} | {row['seed']} | "
            f"{row['steps']} | {row['fail_to_pass_resolved']}/{row['fail_to_pass_total']} | "
            f"{', '.join(row['changed_files']) or '-'} | {len(row['edits_applied'])} | "
            f"{row['edit_refusals']} | {row['finish_refusals']} | "
            f"{', '.join(row['violations']) or '-'} |"
        )
    lines.extend(["", "## Most common edit refusals", ""])
    for reason, count in summary["refusal_reasons"]:
        lines.append(f"- {count}x {reason}")
    lines.extend(["", "## Most common search queries", ""])
    for query, count in summary["queries"]:
        lines.append(f"- {count}x `{query}`")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--gold-patches", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trajectories = _jsonl(Path(args.trajectories))
    gold = json.loads(Path(args.gold_patches).read_text(encoding="utf-8"))
    rows = build_rows(trajectories, gold)
    summary = summarise(rows)
    report = {"schema_version": 1, "kind": "abc-diagnostic-summary", "rows": rows, "summary": summary}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        Path(args.markdown).write_text(markdown(rows, summary), encoding="utf-8")
    print(json.dumps(summary["by_condition"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
