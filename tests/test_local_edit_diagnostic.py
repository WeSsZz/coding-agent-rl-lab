from __future__ import annotations

import json
import unittest

from coding_agent_rl_lab.edit_action_evaluate import _action_from_parsed
from scripts.build_local_edit_diagnostic_contexts import (
    CALLBACK_PATH,
    LocalEditContextBuildError,
    build_contexts,
)


def _step(sequence: int, kind: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "sequence": sequence,
        "action": {"kind": kind, "arguments": arguments},
        "observation": "Updated file." if kind.startswith("replace_") else "observation",
        "terminated": kind == "run_tests",
    }


def _trajectory(state_id: str, edit_count: int) -> dict[str, object]:
    recovery = [_step(0, "read_file", {"path": CALLBACK_PATH})]
    for index in range(edit_count):
        recovery.append(
            _step(
                len(recovery),
                "replace_text",
                {"path": CALLBACK_PATH, "old": f"old-{index}", "new": f"new-{index}"},
            )
        )
    recovery.append(_step(len(recovery), "run_tests", {}))
    return {
        "schema_version": 1,
        "task_id": "getmoto__moto-7607",
        "state_id": state_id,
        "source_session_id": state_id,
        "source_action_count": 1,
        "source_trace_sha256": state_id,
        "base_commit": "base",
        "initial_messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "issue"},
        ],
        "prefix_messages": [
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": "search_text", "arguments": {"query": "callback"}}
                ),
            },
            {"role": "tool", "name": "search_text", "content": "match"},
        ],
        "recovery_steps": recovery,
        "verifier": {"strict_success": True},
    }


class LocalEditDiagnosticTests(unittest.TestCase):
    def test_builds_seven_edit_and_nine_staged_contexts(self) -> None:
        edits, staged = build_contexts(
            [_trajectory("a", 2), _trajectory("b", 2), _trajectory("c", 3)]
        )
        self.assertEqual(len(edits), 7)
        self.assertEqual(len(staged), 9)
        self.assertTrue(all("target_action" not in row["prompt"] for row in edits))
        self.assertTrue(
            all(message.get("tool_calls") for row in staged for message in row["prompt"] if message["role"] == "assistant")
        )

    def test_rejects_non_strict_source(self) -> None:
        rows = [_trajectory("a", 2), _trajectory("b", 2), _trajectory("c", 3)]
        rows[0]["verifier"] = {"strict_success": False}
        with self.assertRaisesRegex(LocalEditContextBuildError, "not verifier-strict"):
            build_contexts(rows)

    def test_parsed_action_accepts_json_arguments(self) -> None:
        action = _action_from_parsed(
            {
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"file.py"}',
                        }
                    }
                ]
            }
        )
        self.assertEqual(action.arguments, {"path": "file.py"})


if __name__ == "__main__":
    unittest.main()
