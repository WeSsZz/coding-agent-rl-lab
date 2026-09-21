"""Pin the diagnostic summariser: refusal classes, arm totals, and the honest denominators.

The summariser is what turns 18 raw trajectories into the numbers a decision is made from, so its
two easy mistakes are pinned here: counting a step that merely *carries* a known test result as a
verifier run, and reporting "refused" as one undifferentiated number when a read-before-edit
refusal and an unparseable-edit refusal answer different questions.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_diagnostic_probe.py"
_spec = importlib.util.spec_from_file_location("summarize_diagnostic_probe", SCRIPT)
assert _spec is not None and _spec.loader is not None
summarize = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = summarize
_spec.loader.exec_module(summarize)

GOLD = {
    "task": """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,2 +1,3 @@
 keep this line
+a distinct added line
"""
}


def _step(kind, arguments, observation, *, test_result=None, prompt_tokens=10, completion_tokens=5):
    return {
        "action": {"kind": kind, "arguments": arguments},
        "observation": observation,
        "test_result": test_result,
        "policy_metadata": {"usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}},
    }


def trajectory(*, condition="A", steps=None, changed=None, resolved=0, total=1, seed=1, repetition=1):
    return {
        "task_id": "task",
        "repetition": repetition,
        "seed": seed,
        "policy": {"metadata": {"diagnostic_condition": condition, "auxiliary_chars": 0}},
        "steps": steps or [],
        "changed_files": changed or [],
        "initial_observation": "Baseline verifier result:\nTests failed (exit=1).",
        "reward": {
            "task_success": False,
            "tests_passed": False,
            "steps": len(steps or []),
            "loop_rejections": 0,
            "violations": [],
        },
        "verifier": {
            "fail_to_pass_total": total,
            "fail_to_pass_resolved": resolved,
            "pass_to_pass_total": 0,
            "pass_to_pass_regressed": 0,
            "failed_nodes": [],
        },
        "training_reward": 0.0,
    }


class RefusalClassTests(unittest.TestCase):
    def test_separates_the_refusals_that_answer_different_questions(self) -> None:
        self.assertEqual(
            summarize.refusal_class(
                "Tool error: replace_lines requires reading the target file first. Read it."
            ),
            "read_before_edit",
        )
        self.assertEqual(
            summarize.refusal_class(
                "Tool error: replace_text failed: edit not applied: it would leave a.py unparseable"
            ),
            "edit_would_not_parse",
        )
        self.assertEqual(
            summarize.refusal_class("Tool error: replace_text requires exactly one match, found 0"),
            "text_not_found_or_ambiguous",
        )
        self.assertEqual(
            summarize.refusal_class("Tool error: do not reread an unchanged file"),
            "repeated_unchanged_read",
        )
        self.assertEqual(
            summarize.refusal_class("Tool error: something new"), "other"
        )


class IndicatorTests(unittest.TestCase):
    def test_a_step_carrying_a_known_test_result_is_not_a_verifier_run(self) -> None:
        steps = [
            _step("read_file", {"path": "pkg/mod.py"}, "1: keep this line", test_result={"passed": False}),
            _step("read_file", {"path": "pkg/mod.py"}, "1: keep this line", test_result={"passed": False}),
            _step("run_tests", {}, "Tests failed", test_result={"passed": False}),
        ]
        row = summarize.build_rows([trajectory(steps=steps)], GOLD)[0]
        self.assertEqual(row["run_tests_actions"], 1)
        self.assertEqual(row["actions_by_kind"]["read_file"], 2)
        self.assertEqual(row["tokens"], 45)

    def test_counts_applied_and_refused_edits_separately(self) -> None:
        steps = [
            _step("replace_text", {"path": "pkg/mod.py", "old": "a", "new": "b"}, "Updated pkg/mod.py."),
            _step("replace_text", {"path": "pkg/mod.py", "old": "a", "new": "b"}, "Tool error: no match"),
        ]
        row = summarize.build_rows([trajectory(steps=steps)], GOLD)[0]
        self.assertEqual(row["edits_applied"], ["pkg/mod.py"])
        self.assertEqual(row["edit_refusals"], 1)
        self.assertTrue(row["applied_an_edit_in_a_repair_file"])
        self.assertEqual(row["refusals_by_class"], {"other": 1})

    def test_records_the_longest_refusal_streak(self) -> None:
        refusal = "Tool error: do not reread an unchanged file"
        steps = [
            _step("read_file", {"path": "pkg/mod.py"}, refusal),
            _step("read_file", {"path": "pkg/mod.py"}, refusal),
            _step("read_file", {"path": "pkg/mod.py"}, "1: keep this line"),
            _step("read_file", {"path": "pkg/mod.py"}, refusal),
        ]
        row = summarize.build_rows([trajectory(steps=steps)], GOLD)[0]
        self.assertEqual(row["longest_refusal_streak"], 2)
        self.assertEqual(row["refused_steps"], 3)

    def test_separates_a_literal_from_its_own_observation_from_reference_text(self) -> None:
        steps = [
            _step("search_text", {"query": "update_item"}, "no matches"),
            _step("search_text", {"query": "a distinct added line"}, "no matches"),
        ]
        initial = "Baseline verifier result:\nassert update_item(x) == 3"
        record = trajectory(steps=steps)
        record["initial_observation"] = initial
        row = summarize.build_rows([record], GOLD)[0]
        self.assertEqual(row["searched_a_literal_from_its_own_observation"], ["update_item"])
        self.assertEqual(row["searched_reference_text"], ["a distinct added line"])


class ArmTests(unittest.TestCase):
    def test_keeps_conditions_and_tasks_apart(self) -> None:
        rows = summarize.build_rows(
            [
                trajectory(condition="A", steps=[_step("read_file", {"path": "pkg/mod.py"}, "1: keep")]),
                trajectory(condition="B", steps=[_step("read_file", {"path": "pkg/mod.py"}, "Tool error: do not reread an unchanged file")]),
                trajectory(condition="C", steps=[_step("read_file", {"path": "other.py"}, "1: keep")]),
            ],
            GOLD,
        )
        summary = summarize.summarise(rows)
        self.assertEqual(set(summary["by_condition"]), {"A", "B", "C"})
        self.assertEqual(summary["by_condition"]["A"]["trials_that_read_a_repair_file"], 1)
        self.assertEqual(summary["by_condition"]["B"]["trials_that_applied_an_edit"], 0)
        self.assertEqual(summary["by_condition"]["C"]["trials_that_read_a_repair_file"], 0)
        self.assertEqual(summary["by_condition"]["B"]["refusals_by_class"], {"repeated_unchanged_read": 1})

    def test_markdown_states_the_assist_in_every_row(self) -> None:
        rows = summarize.build_rows([trajectory(condition="B")], GOLD)
        rendered = summarize.markdown(rows, summarize.summarise(rows))
        self.assertIn("B oracle-file (assisted)", rendered)
        self.assertIn("## Per trial", rendered)


if __name__ == "__main__":
    unittest.main()
