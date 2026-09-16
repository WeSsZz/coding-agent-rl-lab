from __future__ import annotations

import unittest

from coding_agent_rl_lab.sft_semantic_mix import (
    SemanticMixError,
    build_semantic_training_mix,
)
from coding_agent_rl_lab.sft_semantic_recovery import SEMANTIC_RECOVERY_ANSWER_SOURCE
from coding_agent_rl_lab.swe_gym_smoke import pinned_rows_for_task_set


def _row(state: str, sequence: int, kind: str, *, source: str) -> dict[str, object]:
    task_id = pinned_rows_for_task_set("train")[0].instance_id
    return {
        "schema_version": 1,
        "example_id": f"{source}-{state}-{sequence}",
        "task_id": task_id,
        "task_set": "train",
        "contains_answers": True,
        "answer_source": source,
        "target_action": {"kind": kind, "arguments": {}},
        "semantic_recovery": {"state_id": state, "sequence": sequence},
    }


class SemanticMixTests(unittest.TestCase):
    def test_balances_three_states_and_builds_exact_75_25_mix(self) -> None:
        kinds = ["search_text", "read_file", "replace_text", "read_file", "replace_text", "run_tests"]
        recovery = [
            _row(state, sequence, kind, source=SEMANTIC_RECOVERY_ANSWER_SOURCE)
            for state in ("a", "b", "c")
            for sequence, kind in enumerate(kinds)
        ]
        audited = [
            _row("official", sequence, kind, source="official_swe_gym_gold_patch")
            for sequence, kind in enumerate(kinds)
        ]

        mixed, report = build_semantic_training_mix(recovery, audited, recovery_per_state=4)

        self.assertEqual(len(mixed), 16)
        self.assertEqual(report["stage_counts"], {"semantic-recovery": 12, "audited-train-tool": 4})
        self.assertEqual(report["mixture_percent"], {"semantic-recovery": 75, "audited-train-tool": 25})
        self.assertEqual(set(report["state_example_counts"].values()), {4})
        self.assertEqual(len({row["example_id"] for row in mixed}), 16)

    def test_requires_three_real_recovery_states(self) -> None:
        recovery = [
            _row(state, sequence, "run_tests", source=SEMANTIC_RECOVERY_ANSWER_SOURCE)
            for state in ("a", "b")
            for sequence in range(4)
        ]
        audited = [
            _row("official", sequence, "run_tests", source="official_swe_gym_gold_patch")
            for sequence in range(4)
        ]
        with self.assertRaisesRegex(SemanticMixError, "three recovery states"):
            build_semantic_training_mix(recovery, audited)


if __name__ == "__main__":
    unittest.main()
