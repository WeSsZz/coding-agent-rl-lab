from __future__ import annotations

import json
import unittest

from coding_agent_rl_lab.swe_gym_sft import (
    MAX_TARGET_ACTION_CHARS,
    SFTDatasetError,
    build_train_gold_sft_dataset,
    parse_unified_diff,
)
from coding_agent_rl_lab.swe_gym_smoke import (
    PINNED_DEVELOPMENT_ROWS,
    pinned_rows_for_task_set,
)


SOURCE_AND_TEST_PATCH = """diff --git a/moto/service/models.py b/moto/service/models.py
--- a/moto/service/models.py
+++ b/moto/service/models.py
@@ -10,2 +10,2 @@
 def calculate(value):
-    return value
+    return max(0, value)
diff --git a/tests/test_service.py b/tests/test_service.py
--- a/tests/test_service.py
+++ b/tests/test_service.py
@@ -4,2 +4,2 @@
-def test_old():
+def test_new():
     pass
"""


def _row(*, pinned_index: int = 0) -> dict[str, object]:
    pinned = pinned_rows_for_task_set("train")[pinned_index]
    return {
        "instance_id": pinned.instance_id,
        "base_commit": pinned.base_commit,
        "repo": "getmoto/moto",
        "version": "5.0",
        "problem_statement": "Clamp the calculated value.",
        "patch": SOURCE_AND_TEST_PATCH,
        "test_patch": "PRIVATE-VERIFIER-PATCH",
        "FAIL_TO_PASS": json.dumps(["tests/test_service.py::test_new"]),
    }


class SWEGymSFTTests(unittest.TestCase):
    def test_unified_diff_parser_preserves_exact_replace_text(self) -> None:
        hunks = parse_unified_diff(SOURCE_AND_TEST_PATCH)

        self.assertEqual(len(hunks), 2)
        self.assertEqual(hunks[0].new_path, "moto/service/models.py")
        self.assertEqual(
            hunks[0].old_text,
            "def calculate(value):\n    return value\n",
        )
        self.assertEqual(
            hunks[0].new_text,
            "def calculate(value):\n    return max(0, value)\n",
        )

    def test_train_gold_dataset_builds_four_stages_and_skips_tests(self) -> None:
        examples, report = build_train_gold_sft_dataset((_row(),))

        self.assertEqual(len(examples), 4)
        self.assertEqual(
            [example["stage"] for example in examples],
            ["locate", "inspect", "edit", "verify"],
        )
        self.assertEqual(report["task_set"], "train")
        self.assertEqual(report["stage_counts"], {
            "edit": 1,
            "inspect": 1,
            "locate": 1,
            "verify": 1,
        })
        self.assertEqual(report["skipped_hunk_counts"], {"test_file": 1})
        self.assertTrue(report["contains_answers"])
        self.assertEqual(report["max_target_action_chars"], MAX_TARGET_ACTION_CHARS)
        self.assertTrue(all(example["contains_answers"] for example in examples))
        self.assertNotIn("PRIVATE-VERIFIER-PATCH", json.dumps(examples))

        inspect = examples[1]["target_action"]
        self.assertEqual(inspect["kind"], "read_file")
        self.assertGreaterEqual(
            inspect["arguments"]["end_line"] - inspect["arguments"]["start_line"] + 1,
            20,
        )
        edit = examples[2]["target_action"]
        self.assertEqual(edit["kind"], "replace_text")
        self.assertIn("return value", edit["arguments"]["old"])

    def test_edit_history_shows_the_numbered_read_file_observation_the_live_tool_returns(self) -> None:
        examples, _ = build_train_gold_sft_dataset((_row(),))
        edit_example = next(example for example in examples if example["stage"] == "edit")
        payload = json.loads(edit_example["messages"][1]["content"])

        read_observations = [
            entry["observation"]
            for entry in payload["history"]
            if (entry.get("action") or {}).get("kind") == "read_file"
        ]
        self.assertEqual(len(read_observations), 1)
        read_observation = read_observations[0]
        # The policy can only learn to derive an unnumbered `old` if the observation it is shown
        # carries the numbering the live `read_file` returns.
        self.assertRegex(read_observation, r"(?m)^1: def calculate\(value\):")

        old = edit_example["target_action"]["arguments"]["old"]
        self.assertNotRegex(old, r"(?m)^\s*\d+:\s")
        self.assertNotIn("[read_file lines", old)

    def test_edit_history_does_not_present_the_target_old_text_as_an_observation(self) -> None:
        examples, _ = build_train_gold_sft_dataset((_row(),))
        edit_example = next(example for example in examples if example["stage"] == "edit")
        old = edit_example["target_action"]["arguments"]["old"]
        rendered = json.dumps(edit_example["messages"][1]["content"])

        # A history observation identical to `old` is what a warm start learns to copy verbatim.
        self.assertNotIn(json.dumps(old), rendered)

    def test_regression_and_held_out_rows_are_hard_rejected(self) -> None:
        held_out = PINNED_DEVELOPMENT_ROWS[-1]
        row = _row()
        row["instance_id"] = held_out.instance_id
        row["base_commit"] = held_out.base_commit

        with self.assertRaisesRegex(SFTDatasetError, "outside the pinned train split"):
            build_train_gold_sft_dataset((row,))

    def test_duplicate_rows_are_rejected(self) -> None:
        row = _row()

        with self.assertRaisesRegex(SFTDatasetError, "duplicate train row"):
            build_train_gold_sft_dataset((row, dict(row)))

    def test_oversized_edit_targets_are_skipped(self) -> None:
        row = _row()
        oversized = "x" * (MAX_TARGET_ACTION_CHARS + 100)
        row["patch"] = (
            "diff --git a/moto/models.py b/moto/models.py\n"
            "--- a/moto/models.py\n"
            "+++ b/moto/models.py\n"
            "@@ -1 +1 @@\n"
            f"-{oversized}\n"
            "+short\n"
            "diff --git a/moto/other.py b/moto/other.py\n"
            "--- a/moto/other.py\n"
            "+++ b/moto/other.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )

        examples, report = build_train_gold_sft_dataset((row,))

        self.assertEqual(len(examples), 4)
        self.assertEqual(
            report["skipped_hunk_counts"],
            {"oversized_target_action": 1},
        )


if __name__ == "__main__":
    unittest.main()
