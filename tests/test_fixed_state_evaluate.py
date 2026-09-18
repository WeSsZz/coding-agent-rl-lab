from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from coding_agent_rl_lab.fixed_state_evaluate import (
    FixedStateEvaluationError,
    _load_contexts,
    _test_status,
)


def _context(state_id: str) -> dict[str, object]:
    action = {"kind": "search_text", "arguments": {"query": "symbol"}}
    return {
        "schema_version": 1,
        "task_id": "getmoto__moto-7607",
        "state_id": state_id,
        "source_session_id": state_id,
        "used_for_training": False,
        "prefix_action_count": 1,
        "prefix_actions": [action],
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": json.dumps({"name": "search_text", "arguments": {"query": "symbol"}})},
            {"role": "tool", "name": "search_text", "content": "match"},
        ],
    }


class FixedStateEvaluateTests(unittest.TestCase):
    def test_loads_unique_real_state_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "contexts.jsonl")
            path.write_text(
                "".join(json.dumps(_context(str(index))) + "\n" for index in range(4)),
                encoding="utf-8",
            )
            contexts = _load_contexts(path)
        self.assertEqual(len(contexts), 4)

    def test_duplicate_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "contexts.jsonl")
            path.write_text(
                json.dumps(_context("same")) + "\n" + json.dumps(_context("same")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FixedStateEvaluationError, "duplicate"):
                _load_contexts(path)

    def test_verifier_status_ignores_runtime_noise(self) -> None:
        first = "Tests failed (exit=1).\nFAILED tests/x.py::test_a\n"
        second = (
            "Tests failed (exit=1).\n"
            "ERROR moto.module:45 runtime log\n"
            "FAILED tests/x.py::test_a\n"
        )
        self.assertEqual(_test_status(first), _test_status(second))

    def test_verifier_status_preserves_pass_fail_identity(self) -> None:
        failed = "Tests failed (exit=1).\nFAILED tests/x.py::test_a\n"
        passed = "Tests failed (exit=1).\nPASSED tests/x.py::test_a\n"
        self.assertNotEqual(_test_status(failed), _test_status(passed))


if __name__ == "__main__":
    unittest.main()
