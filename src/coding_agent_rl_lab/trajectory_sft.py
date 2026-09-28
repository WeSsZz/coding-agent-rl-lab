"""CPU preparation for native OpenHands trajectories; never executes dataset text.

Keep the legacy synthetic-action pipeline independent. Success is an upstream
label, not a local reproduction. Repository exclusions deliberately exceed the
minimum instance-level split because the source lacks base_commit metadata.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path


class TrajectoryError(ValueError):
    pass


def strip_arrow_nulls(value):
    """Arrow unions add null fields to otherwise distinct JSON tool schemas."""
    if isinstance(value, dict):
        return {k: strip_arrow_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [strip_arrow_nulls(v) for v in value]
    return value


def normalize(row: dict, tools: list, blocked_repos: set[str]) -> list[dict]:
    for key in ("trajectory_id", "instance_id", "repo"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise TrajectoryError("missing_" + key)
    if row["repo"] in blocked_repos:
        raise TrajectoryError("evaluation_repository")
    if strip_arrow_nulls(row.get("tools")) != tools:
        raise TrajectoryError("tool_schema_mismatch")
    if row.get("resolved") != 1 or row.get("exit_status") != "submit":
        raise TrajectoryError("not_successful_submission")
    if not isinstance(row.get("model_patch"), str) or not row["model_patch"].strip():
        raise TrajectoryError("empty_patch")
    messages = copy.deepcopy(row.get("trajectory"))
    if not isinstance(messages, list) or len(messages) < 3:
        raise TrajectoryError("missing_trajectory")
    schemas = {t["function"]["name"]: t["function"] for t in tools}
    pending, seen = {}, set()
    clean = []
    assistant_count = 0
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise TrajectoryError("invalid_message")
        role = msg.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise TrajectoryError("unknown_role")
        if (i == 0 and role != "system") or (i > 0 and role == "system"):
            raise TrajectoryError("system_position")
        if pending and role != "tool":
            raise TrajectoryError("missing_tool_result")
        content = msg.get("content")
        if content is None and role == "assistant":
            content = ""
        if not isinstance(content, str):
            raise TrajectoryError("nontext_content")
        out = {"role": role, "content": content}
        if role == "assistant":
            assistant_count += 1
            calls = msg.get("tool_calls") or []
            if not isinstance(calls, list):
                raise TrajectoryError("invalid_tool_calls")
            for call in calls:
                if not isinstance(call, dict) or call.get("type") != "function":
                    raise TrajectoryError("invalid_tool_call")
                fn = call.get("function", {})
                name, args, cid = fn.get("name"), fn.get("arguments"), call.get("id")
                if name not in schemas:
                    raise TrajectoryError("unknown_tool")
                if not isinstance(cid, str) or not cid or cid in seen:
                    raise TrajectoryError("duplicate_or_missing_call_id")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError as exc:
                        raise TrajectoryError("invalid_arguments_json") from exc
                if not isinstance(args, dict):
                    raise TrajectoryError("arguments_not_object")
                required = schemas[name].get("parameters", {}).get("required", [])
                if any(k not in args for k in required):
                    raise TrajectoryError("missing_required_argument")
                from jsonschema import Draft7Validator
                if not Draft7Validator(schemas[name].get("parameters", {})).is_valid(args):
                    raise TrajectoryError("invalid_argument_schema")
                fn["arguments"] = args
                seen.add(cid)
                pending[cid] = name
            if calls:
                out["tool_calls"] = calls
            elif not content:
                raise TrajectoryError("empty_assistant")
        if role == "tool":
            cid = msg.get("tool_call_id")
            if cid not in pending:
                raise TrajectoryError("orphan_tool_result")
            name = pending.pop(cid)
            if msg.get("name") not in (None, name):
                raise TrajectoryError("tool_name_mismatch")
            out.update(tool_call_id=cid, name=name)
        clean.append(out)
    # OpenHands may end immediately after finish, without a tool observation.
    if pending and (len(pending) != 1 or next(iter(pending.values())) != "finish"):
        raise TrajectoryError("missing_tool_result")
    if not assistant_count or not any(m["role"] == "user" for m in clean):
        raise TrajectoryError("missing_user_or_assistant")
    return clean


def assistant_examples(messages: list[dict]):
    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            yield {"prompt": messages[:i], "completion": [message], "turn": i}


def nonthinking_messages(tokenizer, messages: list[dict], enable_thinking: bool) -> list[dict]:
    """Keep Qwen3's earlier assistant turns aligned with its non-thinking prompt.

    The pinned template emits an empty think block for the last assistant turn,
    but omits it from earlier turns unless ``reasoning_content`` is truthy. A
    newline is truthy to Jinja and strips to empty, so every assistant turn gets
    the same empty block as the generation prompt.
    """
    if enable_thinking or "enable_thinking" not in str(getattr(tokenizer, "chat_template", "")):
        return messages
    normalized = copy.deepcopy(messages)
    for message in normalized:
        if message["role"] == "assistant":
            if str(message.get("reasoning_content") or "").strip():
                raise TrajectoryError("reasoning_content_in_non_thinking")
            message["reasoning_content"] = "\n"
    return normalized


def encode_example(tokenizer, example: dict, tools: list, max_length: int, *, enable_thinking: bool = False) -> dict:
    """Exact token prefix check; never train on observations or truncate actions."""
    if max_length <= 0:
        raise TrajectoryError("invalid_max_length")
    messages = nonthinking_messages(tokenizer, example["prompt"] + example["completion"], enable_thinking)
    prompt_messages = messages[:len(example["prompt"])]
    prompt = tokenizer.apply_chat_template(prompt_messages, tools=tools,
                                           tokenize=False, add_generation_prompt=True,
                                           enable_thinking=enable_thinking)
    full = tokenizer.apply_chat_template(messages,
                                         tools=tools, tokenize=False,
                                         add_generation_prompt=False,
                                         enable_thinking=enable_thinking)
    pids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = tokenizer.encode(full, add_special_tokens=False)
    if ids[:len(pids)] != pids:
        raise TrajectoryError("token_prefix_mismatch")
    if len(ids) <= len(pids):
        raise TrajectoryError("no_supervised_tokens")
    if len(ids) > max_length:
        raise TrajectoryError("over_max_length")
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(pids) + ids[len(pids):]}


def encode_trajectory(tokenizer, messages: list[dict], tools: list, max_length: int, *, enable_thinking: bool = False) -> dict:
    """One intact conversation per sequence, with every assistant turn supervised.

    Prefix token equality prevents template/BPE boundary changes from silently
    moving a label onto a tool observation. No packing, cropping or role guessing.
    """
    messages = nonthinking_messages(tokenizer, messages, enable_thinking)
    full = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                         add_generation_prompt=False,
                                         enable_thinking=enable_thinking)
    if getattr(tokenizer, "is_fast", False):
        batch = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = batch["input_ids"], batch["offset_mapping"]
        if max_length <= 0 or len(ids) > max_length:
            raise TrajectoryError("over_max_length")
        labels = [-100] * len(ids)
        # Map rendered assistant spans onto actual tokens in the one complete
        # sequence. Reject any token crossing a supervision boundary.
        boundaries = {0, len(full)} | {v for pair in offsets for v in pair}
        for example in assistant_examples(messages):
            prompt = tokenizer.apply_chat_template(example["prompt"], tools=tools,
                                                    tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=enable_thinking)
            end_text = tokenizer.apply_chat_template(example["prompt"] + example["completion"],
                                                      tools=tools, tokenize=False, add_generation_prompt=False,
                                                      enable_thinking=enable_thinking)
            if not full.startswith(prompt) or not full.startswith(end_text):
                raise TrajectoryError("trajectory_prefix_mismatch")
            start, end = len(prompt), len(end_text)
            if start >= end or start not in boundaries or end not in boundaries:
                raise TrajectoryError("token_boundary_mismatch")
            for i, (a, b) in enumerate(offsets):
                if a < start < b or a < end < b:
                    raise TrajectoryError("token_boundary_mismatch")
                if start <= a < b <= end:
                    labels[i] = ids[i]
        if all(v == -100 for v in labels):
            raise TrajectoryError("no_supervised_tokens")
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
    ids = tokenizer.encode(full, add_special_tokens=False)
    if max_length <= 0 or len(ids) > max_length:
        raise TrajectoryError("over_max_length")
    labels = [-100] * len(ids)
    for example in assistant_examples(messages):
        encoded = encode_example(tokenizer, example, tools, max_length,
                                 enable_thinking=enable_thinking)
        end = len(encoded["input_ids"])
        if ids[:end] != encoded["input_ids"]:
            raise TrajectoryError("trajectory_prefix_mismatch")
        for i, value in enumerate(encoded["labels"]):
            if value != -100:
                labels[i] = value
    if all(v == -100 for v in labels):
        raise TrajectoryError("no_supervised_tokens")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def pad_training_batch(features: list[dict], pad_token_id: int) -> dict:
    """Right padding is attention-masked and excluded from loss, even for EOS pads."""
    width = max(len(f["input_ids"]) for f in features)
    result = {key: [] for key in ("input_ids", "attention_mask", "labels")}
    for feature in features:
        n = len(feature["input_ids"])
        if not (n == len(feature["attention_mask"]) == len(feature["labels"])):
            raise TrajectoryError("inconsistent_token_lengths")
        for key, pad in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100)):
            result[key].append(feature[key] + [pad] * (width - n))
    return result


def prepare(input_path: Path, split_path: Path, tools_path: Path, output: Path):
    if output.exists():
        raise TrajectoryError("output_already_exists")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    blocked = {r["repo"] for r in split["instances"] if r["split"] != "train"}
    tools = json.loads(tools_path.read_text(encoding="utf-8"))
    accepted, decisions, ids, contents = [], [], set(), set()
    with input_path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TrajectoryError("row_not_object")
            decision = {"line": line_no, "trajectory_id": row.get("trajectory_id"),
                        "instance_id": row.get("instance_id")}
            try:
                messages = normalize(row, tools, blocked)
                digest = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
                if row["trajectory_id"] in ids or digest in contents:
                    raise TrajectoryError("duplicate_trajectory")
                ids.add(row["trajectory_id"])
                contents.add(digest)
                accepted.append({"trajectory_id": row["trajectory_id"],
                                 "instance_id": row["instance_id"], "repo": row["repo"],
                                 "messages": messages, "messages_sha256": digest})
                decision["status"] = "accepted"
            except TrajectoryError as exc:
                decision.update(status="rejected", reason=str(exc))
            decisions.append(decision)
    output.mkdir(parents=True)
    def write_jsonl(name, rows):
        (output / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    write_jsonl("trajectories.jsonl", accepted)
    write_jsonl("decisions.jsonl", decisions)
    report = {"schema": "openhands-trajectory-preparation-v1", "accepted": len(accepted),
              "assistant_examples": sum(len(list(assistant_examples(r["messages"]))) for r in accepted),
              "rejections": dict(Counter(d["reason"] for d in decisions if d["status"] == "rejected")),
              "blocked_repositories": sorted(blocked), "success_evidence": "upstream_label_only",
              "training_ready": False, "tokenizer_checked": False, "runtime_checked": False,
              "hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (input_path, split_path, tools_path, output / "trajectories.jsonl")}}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("manifests/route-v2-task-split.json"))
    parser.add_argument("--tools", type=Path, default=Path("envs/openhands-v0.54-tools.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.input, args.split, args.tools, args.output)
    print(json.dumps(report, indent=2))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
