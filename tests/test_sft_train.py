from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_agent_rl_lab.model_policy import PROMPT_VERSION
from coding_agent_rl_lab.sft_train import (
    SFTTrainingError,
    _metric_number,
    _validate_args,
    build_parser,
    build_token_length_report,
    load_sft_examples,
    prepare_prompt_completion_rows,
)
from coding_agent_rl_lab.sft_semantic_recovery import (
    MIXED_AUDITED_ANSWER_SOURCE,
    SEMANTIC_RECOVERY_ANSWER_SOURCE,
)
from coding_agent_rl_lab.swe_gym_smoke import pinned_rows_for_task_set


def _example() -> dict[str, object]:
    task_id = pinned_rows_for_task_set("train")[0].instance_id
    action = {"kind": "run_tests", "arguments": {}}
    return {
        "schema_version": 1,
        "example_id": "sft-example-1",
        "task_id": task_id,
        "task_set": "train",
        "stage": "verify",
        "source_path": "moto/models.py",
        "hunk_index": 1,
        "prompt_version": PROMPT_VERSION,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": '{"kind":"run_tests","arguments":{}}',
            },
        ],
        "target_action": action,
        "contains_answers": True,
        "answer_source": "official_swe_gym_gold_patch",
    }


def _report() -> dict[str, object]:
    return {
        "dataset_schema": "coding-agent-gold-sft-v1",
        "task_set": "train",
        "task_ids": [pinned_rows_for_task_set("train")[0].instance_id],
        "example_count": 1,
        "contains_answers": True,
        "answer_source": "official_swe_gym_gold_patch",
        "prompt_version": PROMPT_VERSION,
    }


class SFTTrainTests(unittest.TestCase):
    def test_parser_accepts_existing_adapter(self) -> None:
        args = build_parser().parse_args(
            [
                "--model-path",
                "/models/base",
                "--adapter-path",
                "/models/adapter",
                "--dataset",
                "data.jsonl",
                "--dataset-report",
                "report.json",
            ]
        )
        self.assertEqual(args.adapter_path, "/models/adapter")

    def test_load_and_prepare_prompt_completion_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory, "dataset.jsonl")
            report = Path(directory, "report.json")
            dataset.write_text(json.dumps(_example()) + "\n", encoding="utf-8")
            report.write_text(json.dumps(_report()), encoding="utf-8")

            examples, loaded_report = load_sft_examples(dataset, report)
            rows = prepare_prompt_completion_rows(examples)

        self.assertEqual(loaded_report["task_set"], "train")
        self.assertEqual([message["role"] for message in rows[0]["prompt"]], ["system", "user"])
        self.assertEqual(rows[0]["completion"][0]["role"], "assistant")

    def test_non_train_or_unmarked_data_is_rejected(self) -> None:
        example = _example()
        example["task_set"] = "held-out"
        report_payload = _report()
        report_payload["contains_answers"] = False
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory, "dataset.jsonl")
            report = Path(directory, "report.json")
            dataset.write_text(json.dumps(example) + "\n", encoding="utf-8")
            report.write_text(json.dumps(report_payload), encoding="utf-8")

            with self.assertRaisesRegex(SFTTrainingError, "contains_answers=true"):
                load_sft_examples(dataset, report)

    def test_grpo_rows_use_rollout_tools_and_structured_history(self) -> None:
        example = _example()
        example["action_protocol"] = "grpo-bare-json"
        example["messages"] = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": '{"name":"search_text","arguments":{"query":"symbol"}}',
            },
            {"role": "tool", "name": "search_text", "content": "match"},
            {
                "role": "assistant",
                "content": '{"name":"run_tests","arguments":{}}',
            },
        ]
        tool_schemas = [{"type": "function", "function": {"name": "search_text"}}]
        with patch(
            "coding_agent_rl_lab.sft_train._grpo_tool_schemas", return_value=tool_schemas
        ):
            row = prepare_prompt_completion_rows([example])[0]

        self.assertEqual(row["tools"], tool_schemas)
        self.assertNotIn("content", row["prompt"][2])
        self.assertEqual(
            row["prompt"][2]["tool_calls"][0]["function"],
            {"name": "search_text", "arguments": {"query": "symbol"}},
        )
        self.assertEqual(
            row["completion"][0]["content"], '{"name":"run_tests","arguments":{}}'
        )

    def test_assistant_content_must_match_target_action(self) -> None:
        example = _example()
        example["messages"][-1]["content"] = '{"kind":"finish","arguments":{}}'
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory, "dataset.jsonl")
            report = Path(directory, "report.json")
            dataset.write_text(json.dumps(example) + "\n", encoding="utf-8")
            report.write_text(json.dumps(_report()), encoding="utf-8")

            with self.assertRaisesRegex(SFTTrainingError, "disagrees"):
                load_sft_examples(dataset, report)

    def test_mixed_audited_answer_sources_preserve_row_provenance(self) -> None:
        first = _example()
        second = _example()
        second["example_id"] = "semantic-example-2"
        second["answer_source"] = SEMANTIC_RECOVERY_ANSWER_SOURCE
        report_payload = _report()
        report_payload.update(
            {
                "example_count": 2,
                "answer_source": MIXED_AUDITED_ANSWER_SOURCE,
                "answer_sources": [
                    "official_swe_gym_gold_patch",
                    SEMANTIC_RECOVERY_ANSWER_SOURCE,
                ],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory, "dataset.jsonl")
            report = Path(directory, "report.json")
            dataset.write_text(
                json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8"
            )
            report.write_text(json.dumps(report_payload), encoding="utf-8")

            examples, _ = load_sft_examples(dataset, report)

        self.assertEqual(
            {example["answer_source"] for example in examples},
            {"official_swe_gym_gold_patch", SEMANTIC_RECOVERY_ANSWER_SOURCE},
        )

    def test_token_length_report_refuses_to_hide_overflow(self) -> None:
        class Tokenizer:
            def apply_chat_template(
                self, messages, *, tokenize, add_generation_prompt, tools=None
            ):
                self.assertions = getattr(self, "assertions", []) + [
                    (tokenize, add_generation_prompt, tools)
                ]
                return "".join(message["content"] for message in messages)

            def encode(self, text, *, add_special_tokens):
                self.add_special_tokens = add_special_tokens
                return list(range(len(text)))

        rows = prepare_prompt_completion_rows([_example()])
        tokenizer = Tokenizer()
        report = build_token_length_report(tokenizer, rows, max_length=10)

        self.assertEqual(report["example_count"], 1)
        self.assertEqual(report["over_max_length_count"], 1)
        self.assertGreater(report["max_full_tokens"], 10)
        self.assertEqual(tokenizer.assertions, [(False, False, None), (False, True, None)])
        self.assertGreater(report["total_supervised_tokens"], 0)
        self.assertFalse(tokenizer.add_special_tokens)

    def test_training_arguments_must_be_positive(self) -> None:
        args = argparse.Namespace(
            example_limit=None,
            max_length=0,
            max_steps=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            checkpoint_steps=None,
        )
        with self.assertRaisesRegex(SystemExit, "--max-length"):
            _validate_args(args)

    def test_metrics_do_not_invent_missing_values(self) -> None:
        self.assertEqual(_metric_number("0.25"), 0.25)
        self.assertIsNone(_metric_number("missing"))


if __name__ == "__main__":
    unittest.main()
