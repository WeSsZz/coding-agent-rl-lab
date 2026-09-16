from __future__ import annotations

import json
import unittest

from coding_agent_rl_lab.sft_navigation import build_navigation_examples


class NavigationSftTests(unittest.TestCase):
    def test_adds_trace_prefix_when_search_exposes_gold_path(self) -> None:
        path = "src/right.py"
        base = {
            "task_id": "task-1",
            "contains_answers": True,
            "answer_source": "official_swe_gym_gold_patch",
        }
        locate = {
            **base,
            "stage": "locate",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "issue"},
                {"role": "assistant", "content": '{"name":"search_text","arguments":{"query":"src/right.py"}}'},
            ],
            "target_action": {"kind": "search_text", "arguments": {"query": path}},
        }
        read_call = {"name": "read_file", "arguments": {"path": path, "start_line": 1, "end_line": 20}}
        inspect = {
            **base,
            "stage": "inspect",
            "messages": locate["messages"][:2]
            + [locate["messages"][-1], {"role": "tool", "content": f"PATH_MATCH:{path}\n"},
               {"role": "assistant", "content": json.dumps(read_call, separators=(",", ":"))}],
            "target_action": {"kind": "read_file", "arguments": read_call["arguments"]},
        }
        traces = [{
            "task_id": "task-1",
            "path": "/v1/sessions/session-a/actions",
            "request": {"action": {"kind": "search_text", "arguments": {"query": "wrong"}}},
            "response": {"observation": f"SUGGESTED_PATH:{path}\n"},
        }]

        examples, report = build_navigation_examples([locate, inspect], traces)

        self.assertEqual(report["source_counts"], {"gold-navigation": 1, "trace-navigation": 1})
        recovered = next(row for row in examples if row["navigation_source"] == "trace-navigation")
        self.assertEqual(recovered["prefix_actions"], 1)
        self.assertEqual(recovered["target_action"]["arguments"]["path"], path)
        self.assertEqual([message["role"] for message in recovered["messages"]],
                         ["system", "user", "assistant", "tool", "assistant"])

    def test_clones_read_that_exposes_gold_import_as_bridge(self) -> None:
        path = "pkg/service_callback.py"
        read_call = {"name": "read_file", "arguments": {"path": path, "start_line": 1, "end_line": 20}}
        inspect = {
            "task_id": "task-1", "task_set": "train", "stage": "inspect",
            "contains_answers": True, "answer_source": "official_swe_gym_gold_patch",
            "messages": [
                {"role": "system", "content": "system"}, {"role": "user", "content": "issue"},
                {"role": "assistant", "content": "{}"}, {"role": "tool", "content": "match"},
                {"role": "assistant", "content": json.dumps(read_call, separators=(",", ":"))},
            ],
            "target_action": {"kind": "read_file", "arguments": read_call["arguments"]},
            "target_tool_call": read_call,
        }
        traces = [{
            "task_id": "task-1", "path": "/v1/sessions/a/actions",
            "request": {"action": {"kind": "read_file", "arguments": {"path": "pkg/service.py"}}},
            "response": {"observation": "from pkg.service_callback import Callback\n"},
        }]

        examples, report = build_navigation_examples([inspect], traces)

        self.assertEqual(report["source_counts"], {
            "gold-navigation": 1, "trace-bridge": 1, "trace-navigation": 1,
        })
        bridge = next(row for row in examples if row["navigation_source"] == "trace-bridge")
        self.assertEqual(bridge["target_action"]["arguments"]["path"], "pkg/service.py")


if __name__ == "__main__":
    unittest.main()
