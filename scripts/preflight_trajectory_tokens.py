"""Inspect real tokenizer lengths/masks on prepared OpenHands trajectories, CPU only."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from coding_agent_rl_lab.trajectory_sft import assistant_examples, encode_example, encode_trajectory, TrajectoryError


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, help="pinned upstream inventory; defaults to tokenizer/inventory.json")
    parser.add_argument("--model-manifest", type=Path, help="pinned model selection; defaults to route v2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--complete-only", action="store_true", help="Check and mask complete sequences; never retain early turns of an overlong trajectory")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite output")
    root = Path(__file__).resolve().parents[1]
    from model_contract import selected_model
    manifest = selected_model(args.model_manifest or root / "manifests/route-v2-model.json")
    for filename, key in [("config.json", "config_json_sha256"),
                          ("tokenizer_config.json", "tokenizer_config_json_sha256")]:
        require(hashlib.sha256((args.tokenizer / filename).read_bytes()).hexdigest() == manifest[key], filename)
    inventory = json.loads((args.inventory or args.tokenizer / "inventory.json").read_text())
    require(inventory["id"] == manifest["id"] and inventory["sha"] == manifest["revision"], "tokenizer revision mismatch")
    blob = (args.tokenizer / "tokenizer.json").read_bytes()
    entry = next(e for e in inventory["siblings"] if e["rfilename"] == "tokenizer.json")
    if entry.get("lfs"):
        require(hashlib.sha256(blob).hexdigest() == entry["lfs"]["sha256"], "tokenizer hash mismatch")
    else:
        require(hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest() == entry["blobId"], "tokenizer hash mismatch")
    report = json.loads((args.prepared / "report.json").read_text())
    for path, expected in report["hashes"].items():
        # Resolve artifacts recorded on this host; cross-host bundles must regenerate reports.
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, "artifact hash mismatch: " + path)
    import transformers
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    config = json.loads((args.tokenizer / "config.json").read_text(encoding="utf-8"))
    require(1 < args.max_length <= config["max_position_embeddings"], "length outside model context")
    require(tokenizer.chat_template == json.loads((args.tokenizer / "tokenizer_config.json").read_text())["chat_template"], "template mismatch")
    tools = json.loads((root / "envs/openhands-v0.54-tools.json").read_text())
    lengths, targets, decisions, reasons = [], [], [], Counter()
    with (args.prepared / "trajectories.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            examples = [{"turn": "complete"}] if args.complete_only else assistant_examples(row["messages"])
            for example in examples:
                decision = {"trajectory_id": row["trajectory_id"], "turn": example["turn"]}
                try:
                    encoded = (encode_trajectory(tokenizer, row["messages"], tools, args.max_length)
                               if args.complete_only else encode_example(tokenizer, example, tools, args.max_length))
                    lengths.append(len(encoded["input_ids"]))
                    targets.append(sum(t != -100 for t in encoded["labels"]))
                    decision.update(status="accepted", tokens=lengths[-1], supervised_tokens=targets[-1])
                except TrajectoryError as exc:
                    reasons[str(exc)] += 1
                    decision.update(status="rejected", reason=str(exc))
                decisions.append(decision)
    groups = defaultdict(list)
    for decision in decisions:
        groups[decision["trajectory_id"]].append(decision)
    result = {"model": manifest["id"], "revision": manifest["revision"],
              "unit": "complete_trajectory" if args.complete_only else "assistant_turn",
              "tokenizer_hashes": {name: hashlib.sha256((args.tokenizer / name).read_bytes()).hexdigest()
                                   for name in ("tokenizer.json", "tokenizer_config.json", "config.json")},
              "transformers_version": transformers.__version__, "max_length": args.max_length,
              "accepted_examples": len(lengths), "rejections": dict(reasons),
              "min_tokens": min(lengths, default=0), "max_tokens": max(lengths, default=0),
              "total_supervised_tokens": sum(targets), "min_supervised_tokens": min(targets, default=0),
              "trajectories_checked": len(groups),
              "complete_trajectories_within_limit": sum(
                  all(d["status"] == "accepted" for d in group) for group in groups.values()),
              "training_ready": False, "runtime_checked": False,
              "prepared_report_sha256": hashlib.sha256((args.prepared / "report.json").read_bytes()).hexdigest(),
              "decisions": decisions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "decisions"}, indent=2))
    if not lengths or any(k != "over_max_length" for k in reasons):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
