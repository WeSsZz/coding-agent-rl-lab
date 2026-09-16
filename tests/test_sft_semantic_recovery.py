from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from coding_agent_rl_lab.sft_semantic_recovery import (
    SEMANTIC_RECOVERY_ANSWER_SOURCE,
    SemanticRecoveryError,
    build_semantic_recovery_examples,
    main,
)
from coding_agent_rl_lab.swe_gym_smoke import pinned_rows_for_task_set


def _trajectory(state_id: str = "state-1") -> dict[str, object]:
    task_id = pinned_rows_for_task_set("train")[0].instance_id
    search = {"kind": "search_text", "arguments": {"query": "Callback"}}
    read = {"kind": "read_file", "arguments": {"path": "moto/right.py"}}
    tests = {"kind": "run_tests", "arguments": {}}
    return {
        "schema_version": 1,
        "task_id": task_id,
        "task_set": "train",
        "state_id": state_id,
        "source_session_id": f"session-{state_id}",
        "source_action_count": 1,
        "source_trace_sha256": "a" * 64,
        "base_commit": "b" * 40,
        "teacher_source": "official train gold plus verified replay",
        "initial_messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "issue"},
        ],
        "prefix_messages": [
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": search["kind"], "arguments": search["arguments"]},
                    separators=(",", ":"),
                ),
            },
            {"role": "tool", "name": "search_text", "content": "PATH_MATCH:moto/right.py"},
        ],
        "recovery_steps": [
            {"action": read, "observation": "source", "terminated": False},
            {"action": tests, "observation": "Tests passed", "terminated": True},
        ],
        "verifier": {
            "strict_success": True,
            "baseline_failure_count": 1,
            "final_failure_count": 0,
            "new_failure_count": 0,
            "violations": [],
        },
    }


class SemanticRecoveryTests(unittest.TestCase):
    def test_builds_completion_examples_from_distinct_verified_states(self) -> None:
        trajectories = [_trajectory(f"state-{index}") for index in range(3)]

        examples, report = build_semantic_recovery_examples(trajectories)

        self.assertEqual(len(examples), 6)
        self.assertEqual(report["state_count"], 3)
        self.assertEqual(report["answer_source"], SEMANTIC_RECOVERY_ANSWER_SOURCE)
        self.assertEqual([m["role"] for m in examples[0]["messages"]], [
            "system", "user", "assistant", "tool", "assistant"
        ])
        self.assertEqual(examples[1]["target_action"]["kind"], "run_tests")

    def test_rejects_cross_split_and_unverified_trajectory(self) -> None:
        cross_split = _trajectory()
        cross_split["task_set"] = "held-out"
        with self.assertRaisesRegex(SemanticRecoveryError, "outside the train split"):
            build_semantic_recovery_examples([cross_split], minimum_state_count=1)

        unverified = _trajectory()
        unverified["verifier"]["final_failure_count"] = 1
        with self.assertRaisesRegex(SemanticRecoveryError, "not a verified strict recovery"):
            build_semantic_recovery_examples([unverified], minimum_state_count=1)

    def test_rejects_incomplete_observation_and_missing_target(self) -> None:
        incomplete = _trajectory()
        incomplete["recovery_steps"][0]["observation"] = ""
        with self.assertRaisesRegex(SemanticRecoveryError, "lacks a real observation"):
            build_semantic_recovery_examples([incomplete], minimum_state_count=1)

        missing = _trajectory()
        missing["recovery_steps"][0]["action"] = {
            "kind": "read_file",
            "arguments": {},
        }
        with self.assertRaisesRegex(SemanticRecoveryError, "incomplete 'read_file'"):
            build_semantic_recovery_examples([missing], minimum_state_count=1)

    def test_state_identity_keeps_repeated_reads_after_a_change(self) -> None:
        trajectory = _trajectory()
        read = trajectory["recovery_steps"][0]
        edit = {
            "action": {
                "kind": "replace_text",
                "arguments": {"path": "moto/right.py", "old": "before", "new": "after"},
            },
            "observation": "Updated moto/right.py.",
            "terminated": False,
        }
        trajectory["recovery_steps"] = [read, edit, copy.deepcopy(read), trajectory["recovery_steps"][1]]

        examples, _ = build_semantic_recovery_examples([trajectory], minimum_state_count=1)

        self.assertEqual([row["target_action"]["kind"] for row in examples], [
            "read_file", "replace_text", "read_file", "run_tests"
        ])
        self.assertEqual(len({row["example_id"] for row in examples}), 4)

    def test_cli_refuses_to_overwrite_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            output = root / "output.jsonl"
            report = root / "report.json"
            source.write_text(json.dumps(_trajectory()) + "\n", encoding="utf-8")
            output.write_text("existing", encoding="utf-8")
            with mock.patch(
                "sys.argv",
                [
                    "sft_semantic_recovery",
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                    "--minimum-state-count",
                    "1",
                ],
            ):
                with self.assertRaisesRegex(SemanticRecoveryError, "refusing to overwrite"):
                    main()


if __name__ == "__main__":
    unittest.main()
