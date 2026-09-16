from __future__ import annotations

import json
import unittest

from coding_agent_rl_lab.sft_cumulative import build_cumulative_examples


def _verify_row(hunk: int, read_line: int) -> dict[str, object]:
    calls = [
        ({"name": "search_text", "arguments": {"query": "src/model.py"}}, "found"),
        (
            {"name": "read_file", "arguments": {"path": "src/model.py", "start_line": read_line, "end_line": read_line}},
            "old",
        ),
        (
            {"name": "replace_text", "arguments": {"path": "src/model.py", "old": "old", "new": f"new-{hunk}"}},
            "updated",
        ),
    ]
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    for call, observation in calls:
        messages.extend(
            (
                {"role": "assistant", "content": json.dumps(call, separators=(",", ":"))},
                {"role": "tool", "name": call["name"], "content": observation},
            )
        )
    run_tests = {"name": "run_tests", "arguments": {}}
    messages.append({"role": "assistant", "content": json.dumps(run_tests, separators=(",", ":"))})
    return {
        "task_id": "getmoto__moto-7514",
        "stage": "verify",
        "hunk_index": hunk,
        "messages": messages,
    }


class CumulativeSFTTests(unittest.TestCase):
    def test_combines_hunks_without_repeating_search_or_intermediate_tests(self) -> None:
        rows = [_verify_row(1, 10), _verify_row(2, 20)]
        examples, report = build_cumulative_examples(rows, {"example_count": 8})
        names = [example["target_action"]["kind"] for example in examples]

        self.assertEqual(names, ["search_text", "read_file", "replace_text", "read_file", "replace_text", "run_tests"])
        self.assertEqual(report["example_count"], 6)
        self.assertEqual(report["trajectory_layout"], "cumulative-hunks-single-final-verifier")
        self.assertEqual(
            [message["role"] for message in examples[-1]["messages"]],
            ["system", "user"] + ["assistant", "tool"] * 5 + ["assistant"],
        )


if __name__ == "__main__":
    unittest.main()
