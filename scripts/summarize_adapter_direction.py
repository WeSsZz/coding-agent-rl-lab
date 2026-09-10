"""Recompute a paired direction check from complete, answer-free reward audits."""
import argparse
import json
from pathlib import Path

from coding_agent_rl_lab.grpo_evaluate import summarize


def read_run(directory):
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if not report["run_complete"] or not report["adapter_unchanged"] or report["optimizer_steps"] != 0:
        raise ValueError("comparison requires complete frozen-adapter evaluations")
    records = {"train": [], "regression": []}
    for task in report["tasks"]:
        audit = directory / (task["task_id"] + "-reward-audit.jsonl")
        trials = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(trials) != report["budget"]["num_generations"] or any(r["task_id"] != task["task_id"] for r in trials):
            raise ValueError("incomplete or mismatched audit")
        task["summary"] = summarize(trials)
        records[task["split"]].extend(trials)
    report["summary"] = {split: summarize(trials) for split, trials in records.items()}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before, after = read_run(args.before), read_run(args.after)
    for key in ("model_path", "prompt_rows_sha256", "seed", "budget", "planned_trial_count", "versions"):
        if before[key] != after[key]:
            raise ValueError(f"unmatched comparison field: {key}")
    task_keys = lambda report: [(t["task_id"], t["split"], t["seed"]) for t in report["tasks"]]
    if task_keys(before) != task_keys(after):
        raise ValueError("task order, split, or seed differs")
    paired = []
    for first, second in zip(before["tasks"], after["tasks"], strict=True):
        paired.append({"task_id": first["task_id"], "split": first["split"], "seed": first["seed"],
                       "before": first["summary"], "after": second["summary"]})
    result = {
        "schema_version": 1, "training_performed": False,
        "before_adapter_sha256": before["adapter_sha256"],
        "after_adapter_sha256": after["adapter_sha256"],
        "budget": before["budget"], "versions": before["versions"],
        "summary": {split: {"before": before["summary"][split], "after": after["summary"][split]}
                    for split in ("train", "regression")},
        "tasks": paired,
        "completed_evaluation_seconds": {
            "before": sum(t["elapsed_seconds"] for t in before["tasks"]),
            "after": sum(t["elapsed_seconds"] for t in after["tasks"]),
        },
        "limitations": ["Two samples per task: exploratory, not a significance claim.",
                        "Unknown final failure counts are not evidence of resolved tests.",
                        "Interrupted infrastructure attempts are excluded; complete task checkpoints are retained.",
                        "Failure-count improvement requires a valid patch and no violations; audits do not retain full test output."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
