from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from coding_agent_rl_lab.sft_grpo import (
    GRPO_ACTION_PROTOCOL,
    GRPO_SFT_PROMPT_VERSION,
    convert_sft_dataset,
)
from coding_agent_rl_lab.sft_train import load_sft_examples
from coding_agent_rl_lab.swe_gym_smoke import pinned_rows_for_task_set


class GRPOSFTTests(unittest.TestCase):
    def test_conversion_preserves_train_boundary_and_changes_tool_protocol(self) -> None:
        task_id = pinned_rows_for_task_set("train")[0].instance_id
        example = {
            "schema_version": 1,
            "example_id": "source-1",
            "task_id": task_id,
            "task_set": "train",
            "stage": "edit",
            "source_path": "moto/models.py",
            "hunk_index": 1,
            "prompt_version": "coding-tools-json-v12",
            "messages": [
                {"role": "system", "content": "old"},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "history": [
                                {
                                    "action": {
                                        "kind": "read_file",
                                        "arguments": {"path": "moto/models.py"},
                                    }
                                }
                            ]
                        }
                    ),
                },
                {
                    "role": "assistant",
                    "content": '{"kind":"replace_text","arguments":{"path":"moto/models.py","old":"a","new":"b"}}',
                },
            ],
            "target_action": {
                "kind": "replace_text",
                "arguments": {"path": "moto/models.py", "old": "a", "new": "b"},
            },
            "contains_answers": True,
            "answer_source": "official_swe_gym_gold_patch",
        }
        report = {
            "dataset_schema": "coding-agent-gold-sft-v1",
            "task_set": "train",
            "task_ids": [task_id],
            "example_count": 1,
            "contains_answers": True,
            "answer_source": "official_swe_gym_gold_patch",
            "prompt_version": "coding-tools-json-v12",
        }

        converted, converted_report = convert_sft_dataset([example], report)

        self.assertEqual(converted_report["action_protocol"], GRPO_ACTION_PROTOCOL)
        self.assertEqual(converted_report["prompt_version"], GRPO_SFT_PROMPT_VERSION)
        self.assertEqual(converted[0]["target_tool_call"]["name"], "replace_text")
        self.assertNotIn('"kind"', converted[0]["messages"][-1]["content"])
        user = json.loads(converted[0]["messages"][1]["content"])
        self.assertEqual(user["history"][0]["action"]["name"], "read_file")

        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory, "dataset.jsonl")
            report_path = Path(directory, "report.json")
            dataset_path.write_text(json.dumps(converted[0]) + "\n", encoding="utf-8")
            report_path.write_text(json.dumps(converted_report), encoding="utf-8")
            loaded, loaded_report = load_sft_examples(dataset_path, report_path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded_report["action_protocol"], GRPO_ACTION_PROTOCOL)


if __name__ == "__main__":
    unittest.main()
