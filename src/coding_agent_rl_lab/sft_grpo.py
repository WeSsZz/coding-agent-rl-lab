from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import AgentAction
from .grpo_environment import _GRPO_SYSTEM_PROMPT
from .grpo_train import bare_json_system_prompt
from .swe_gym_smoke import pinned_rows_for_task_set


GRPO_SFT_SCHEMA = "coding-agent-grpo-gold-sft-v2"
GRPO_SFT_PROMPT_VERSION = "grpo-tools-dynamic-bare-json-v2"
GRPO_ACTION_PROTOCOL = "grpo-bare-json"

_SYSTEM_PROMPT = bare_json_system_prompt(_GRPO_SYSTEM_PROMPT)


class GRPOSFTConversionError(ValueError):
    pass


def convert_sft_dataset(
    examples: list[dict[str, Any]],
    source_report: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_source_report(source_report)
    allowed = {item.instance_id for item in pinned_rows_for_task_set("train")}
    converted: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        if (
            not isinstance(example, dict)
            or example.get("task_set") != "train"
            or example.get("task_id") not in allowed
            or example.get("contains_answers") is not True
            or example.get("answer_source") != "official_swe_gym_gold_patch"
        ):
            raise GRPOSFTConversionError(f"source SFT row {index} is not audited train-only data")
        messages = example.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise GRPOSFTConversionError(f"source SFT row {index} has invalid messages")
        try:
            action = AgentAction.from_dict(example["target_action"])
            user_payload = json.loads(messages[1]["content"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise GRPOSFTConversionError(f"source SFT row {index} has invalid action context") from exc
        if not isinstance(user_payload, dict):
            raise GRPOSFTConversionError(f"source SFT row {index} user context must be an object")
        tool_call = {"name": action.kind.value, "arguments": action.arguments}
        target_text = json.dumps(tool_call, ensure_ascii=False, separators=(",", ":"))
        dynamic_messages = _dynamic_tool_messages(user_payload, target_text)
        identity = f"{example.get('example_id', index)}\0{target_text}"
        converted.append(
            {
                **example,
                "example_id": "grpo-sft-"
                + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
                "prompt_version": GRPO_SFT_PROMPT_VERSION,
                "action_protocol": GRPO_ACTION_PROTOCOL,
                "messages": dynamic_messages,
                "target_tool_call": tool_call,
            }
        )
    report = {
        **source_report,
        "dataset_schema": GRPO_SFT_SCHEMA,
        "prompt_version": GRPO_SFT_PROMPT_VERSION,
        "action_protocol": GRPO_ACTION_PROTOCOL,
        "example_count": len(converted),
        "source_dataset_schema": source_report["dataset_schema"],
        "source_prompt_version": source_report["prompt_version"],
        "training_performed": False,
        "intended_use": "train-split-only-grpo-tool-warm-start",
    }
    return converted, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert audited train-only gold SFT data to GRPO bare-JSON tool calls"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-report", required=True)
    parser.add_argument(
        "--output",
        default="work/private/swe-gym-train-gold-grpo-sft-v2.jsonl",
    )
    parser.add_argument(
        "--report",
        default="work/private/swe-gym-train-gold-grpo-sft-v2-report.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[2]
    examples = _load_jsonl(root / args.input)
    source_report = _load_object(root / args.input_report)
    converted, report = convert_sft_dataset(examples, source_report)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in converted),
        encoding="utf-8",
    )
    report_path = root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _validate_source_report(report: dict[str, Any]) -> None:
    if (
        report.get("dataset_schema") != "coding-agent-gold-sft-v1"
        or report.get("task_set") != "train"
        or report.get("contains_answers") is not True
        or report.get("answer_source") != "official_swe_gym_gold_patch"
    ):
        raise GRPOSFTConversionError("source report is not audited train-only gold SFT data")


def _dynamic_tool_messages(user_payload: dict[str, Any], target_text: str) -> list[dict[str, Any]]:
    required = ("task_id", "issue", "base_commit", "repository", "initial_observation", "history")
    if any(key not in user_payload for key in required):
        raise GRPOSFTConversionError("source SFT user context is incomplete")
    if not isinstance(user_payload["initial_observation"], str) or not isinstance(
        user_payload["history"], list
    ):
        raise GRPOSFTConversionError("source SFT user context has invalid observation history")
    task_payload = {
        key: user_payload[key]
        for key in ("task_id", "issue", "base_commit", "repository")
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(task_payload, ensure_ascii=False)
            + user_payload["initial_observation"],
        },
    ]
    for step in user_payload["history"]:
        if not isinstance(step, dict) or not isinstance(step.get("observation"), str):
            raise GRPOSFTConversionError("source SFT history step is invalid")
        try:
            action = AgentAction.from_dict(step["action"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GRPOSFTConversionError("source SFT history action is invalid") from exc
        tool_call = {"name": action.kind.value, "arguments": action.arguments}
        messages.extend(
            (
                {
                    "role": "assistant",
                    "content": json.dumps(tool_call, ensure_ascii=False, separators=(",", ":")),
                },
                {
                    "role": "tool",
                    "name": action.kind.value,
                    "content": step["observation"],
                },
            )
        )
    messages.append({"role": "assistant", "content": target_text})
    return messages


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GRPOSFTConversionError(f"invalid source JSON on row {line_number}") from exc
        if not isinstance(value, dict):
            raise GRPOSFTConversionError(f"source row {line_number} must be an object")
        rows.append(value)
    if not rows:
        raise GRPOSFTConversionError("source dataset must not be empty")
    return rows


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GRPOSFTConversionError("source report is not valid JSON") from exc
    if not isinstance(value, dict):
        raise GRPOSFTConversionError("source report must be an object")
    return value


if __name__ == "__main__":
    main()
