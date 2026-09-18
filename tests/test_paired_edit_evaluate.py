from __future__ import annotations

import unittest

from coding_agent_rl_lab.paired_edit_evaluate import (
    _condition_summary,
    _special_token_stats,
    _termination_reason,
)


class _Tokenizer:
    @staticmethod
    def convert_ids_to_tokens(token_id: int) -> str:
        return {1: "<eos>", 2: "<special>"}.get(token_id, f"tok-{token_id}")


class PairedEditEvaluateTests(unittest.TestCase):
    def test_termination_distinguishes_eos_and_length(self) -> None:
        self.assertEqual(
            _termination_reason([4, 1], eos_ids={1}, max_new_tokens=4), "eos_token"
        )
        self.assertEqual(
            _termination_reason([4, 4, 4, 4], eos_ids={1}, max_new_tokens=4),
            "max_new_tokens",
        )

    def test_special_token_stats_preserve_ids_and_counts(self) -> None:
        result = _special_token_stats(
            [2, 2, 9, 1],
            all_special_ids={1, 2},
            eos_ids={1},
            tokenizer=_Tokenizer(),
        )
        self.assertEqual(result["special_token_count"], 3)
        self.assertEqual(result["nonterminal_special_token_count"], 2)
        self.assertEqual(result["counts"][1]["token"], "<special>")

    def test_condition_summary_uses_suffix_for_semantics(self) -> None:
        state = {
            "generated_action": {"kind": "replace_text", "arguments": {}},
            "immediate": {
                "first_edit_applied": True,
                "strict_success": False,
                "resolved_failure_count": 0,
            },
            "with_teacher_suffix": {"strict_success": True},
            "execution_semantic_success": True,
            "special_token_degeneration": False,
            "termination_reason": "eos_token",
            "exact_teacher_action_match": False,
        }
        summary = _condition_summary([state])
        self.assertEqual(summary["immediate_failure_reduction_count"], 0)
        self.assertEqual(summary["execution_semantic_success_count"], 1)


if __name__ == "__main__":
    unittest.main()
