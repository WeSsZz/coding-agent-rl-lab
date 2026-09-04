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
