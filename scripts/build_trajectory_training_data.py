"""Export complete, token-checked conversations; reject the entire overlong trajectory."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from coding_agent_rl_lab.trajectory_sft import encode_trajectory, TrajectoryError


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, help="pinned model selection; defaults to route v2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int,
                        help="Recheck all complete trajectories at this length, bounded by pinned model context")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output already exists")
    root = Path(__file__).resolve().parents[1]
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    if preflight["prepared_report_sha256"] != sha(args.prepared / "report.json"):
        raise SystemExit("preflight does not match preparation")
    report = json.loads((args.prepared / "report.json").read_text(encoding="utf-8"))
    for path, expected in report["hashes"].items():
        if sha(Path(path)) != expected:
            raise SystemExit("preparation artifact changed: " + path)
    from model_contract import selected_model
    model = selected_model(args.model_manifest or root / "manifests/route-v2-model.json")
    if preflight["model"] != model["id"] or preflight["revision"] != model["revision"]:
        raise SystemExit("model mismatch")
    # Tokenizer identity is bound by preflight; verify each local file again.
    for filename, digest in preflight["tokenizer_hashes"].items():
        if sha(args.tokenizer / filename) != digest:
            raise SystemExit("tokenizer changed: " + filename)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    max_length = args.max_length or preflight["max_length"]
    config = json.loads((args.tokenizer / "config.json").read_text(encoding="utf-8"))
    if not 1 < max_length <= config["max_position_embeddings"]:
        raise SystemExit("length outside pinned model context")
    tools = json.loads((root / "envs/openhands-v0.54-tools.json").read_text(encoding="utf-8"))
    decisions, seen, encoded_rows = [], set(), []
    for line in (args.prepared / "trajectories.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        decision = {k: row[k] for k in ("trajectory_id", "instance_id", "repo")}
        try:
            if row["instance_id"] in seen:
                raise TrajectoryError("duplicate_instance")
            encoded = encode_trajectory(tokenizer, row["messages"], tools, max_length)
            seen.add(row["instance_id"])
            decision.update(status="accepted", tokens=len(encoded["input_ids"]),
                            supervised_tokens=sum(v != -100 for v in encoded["labels"]))
            encoded_rows.append({**decision, **encoded})
        except TrajectoryError as exc:
            decision.update(status="rejected", reason=str(exc))
        decisions.append(decision)
    args.output.mkdir(parents=True)
    data_path = args.output / "train.jsonl"
    data_path.write_text("".join(json.dumps(r) + "\n" for r in encoded_rows), encoding="utf-8")
    result = {"schema": "complete-trajectory-tokens-v1", "model": model["id"], "revision": model["revision"],
              "max_length": max_length, "data_sha256": sha(data_path),
              "tokenizer_hashes": preflight["tokenizer_hashes"],
              "tools_sha256": sha(root / "envs/openhands-v0.54-tools.json"),
              "split_sha256": sha(root / "manifests/route-v2-task-split.json"),
              "preflight_sha256": sha(args.preflight), "trajectory_count": len(encoded_rows),
              "total_tokens": sum(len(r["input_ids"]) for r in encoded_rows),
              "total_supervised_tokens": sum(r["supervised_tokens"] for r in encoded_rows),
              "rejections": dict(Counter(d["reason"] for d in decisions if d["status"] == "rejected")),
              "decisions": decisions, "training_ready": False, "runtime_checked": False,
              "purpose": "short-trajectory SFT smoke candidate; upstream success labels, no local replay"}
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "decisions"}, indent=2))
    if not encoded_rows:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
