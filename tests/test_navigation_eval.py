from __future__ import annotations

import unittest

from coding_agent_rl_lab.navigation_eval import summarize_navigation


class NavigationEvalTests(unittest.TestCase):
    def test_counts_gold_read_within_action_budget(self) -> None:
        gold = [{
            "task_id": "task-1", "stage": "inspect",
            "target_action": {"kind": "read_file", "arguments": {"path": "src/right.py"}},
        }]
        traces = [
            {"task_id": "task-1", "path": "/v1/sessions/a/actions",
             "request": {"action": {"kind": "search_text", "arguments": {"query": "right"}}}},
            {"task_id": "task-1", "path": "/v1/sessions/a/actions",
             "request": {"action": {"kind": "read_file", "arguments": {"path": "src/right.py"}}}},
            {"task_id": "task-1", "path": "/v1/sessions/b/actions",
             "request": {"action": {"kind": "read_file", "arguments": {"path": "src/wrong.py"}}}},
        ]

        report = summarize_navigation(gold, traces, max_actions=2)

        self.assertEqual(report["trial_count"], 2)
        self.assertEqual(report["target_file_hits"], 1)
        self.assertEqual(report["tasks"]["task-1"]["target_file_hit_rate"], 0.5)

    def test_uses_navigation_sft_rows_as_gold_paths(self) -> None:
        gold = [{
            "task_id": "task-1", "stage": "navigation",
            "source_path": "src/right.py",
            "target_action": {"kind": "read_file", "arguments": {"path": "src/bridge.py"}},
        }]
        traces = [{
            "task_id": "task-1", "path": "/v1/sessions/a/actions",
            "request": {"action": {
                "kind": "read_file", "arguments": {"path": "src/right.py"},
            }},
        }]

        report = summarize_navigation(gold, traces)

        self.assertEqual(report["target_file_hits"], 1)


if __name__ == "__main__":
    unittest.main()
