from __future__ import annotations

import json
import unittest

from coding_agent_rl_lab.sft_recovery import build_suggested_path_examples


class SuggestedPathRecoveryTests(unittest.TestCase):
    def test_turns_a_failed_search_suggestion_into_a_read_target(self) -> None:
        read = {"name": "read_file", "arguments": {"path": "src/right.py", "start_line": 1, "end_line": 20}}
        gold = [{
            "task_id": "task-1",
            "stage": "cumulative",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "issue"},
                {"role": "assistant", "content": json.dumps(read, separators=(",", ":"))},
            ],
            "target_action": {"kind": "read_file", "arguments": read["arguments"]},
            "target_tool_call": read,
        }]
        traces = [{
            "task_id": "task-1",
            "request": {"action": {"kind": "search_text", "arguments": {"query": "src/wrong.py"}}},
            "response": {"observation": "No exact matches\nSUGGESTED_PATH:src/right.py\n"},
        }]

        examples = build_suggested_path_examples(
            gold, traces, task_id="task-1", suggested_path="src/right.py"
        )

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["stage"], "recovery")
        self.assertEqual([m["role"] for m in examples[0]["messages"]], ["system", "user", "assistant", "tool", "assistant"])
        self.assertEqual(examples[0]["target_action"]["arguments"]["path"], "src/right.py")


if __name__ == "__main__":
    unittest.main()
