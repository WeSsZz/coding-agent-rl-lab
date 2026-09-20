from __future__ import annotations

import json
import unittest

from coding_agent_rl_lab.swe_gym_sft import (
    MAX_TARGET_ACTION_CHARS,
    SFTDatasetError,
    _require_no_gold_leak,
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

    def test_initial_observation_names_the_assertion_a_failing_test_makes(self) -> None:
        # The live observation carries `[failing statement] <path>:<line>: <assert ...>`; a
        # training row that stops at the test name teaches the policy to answer a failure by
        # naming a file instead of reading what the failure says. This reproduces the shape of
        # the held-out row for `getmoto__moto-7393`, whose real failing line is the first check of
        # what the unimplemented route returned.
        row = _row()
        row["test_patch"] = (
            "diff --git a/tests/test_core/test_config.py b/tests/test_core/test_config.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/tests/test_core/test_config.py\n"
            "@@ -0,0 +1,8 @@\n"
            "+def test_change_configuration_using_api() -> None:\n"
            '+    assert default_user_config["batch"] == {"use_docker": True}\n'
            "+\n"
            '+    resp = requests.get("http://motoapi.amazonaws.com/moto-api/config")\n'
            '+    assert resp.json()["batch"] == {"use_docker": True}\n'
            '+    assert resp.json()["lambda"] == {"use_docker": True}\n'
            "+\n"
            '+    assert resp.status_code == 200\n'
        )

        examples, _ = build_train_gold_sft_dataset((row,))
        payload = json.loads(examples[0]["messages"][1]["content"])

        self.assertIn(
            "[failing statement] tests/test_core/test_config.py:5: "
            'assert resp.json()["batch"] == {"use_docker": True}',
            payload["initial_observation"],
        )
        # The statement comes from the test the verifier runs, never from the gold patch.
        self.assertNotIn("default_user_config", payload["initial_observation"])

    def test_initial_observation_still_names_a_failure_without_an_asserted_response(self) -> None:
        row = _row()
        row["test_patch"] = (
            "diff --git a/tests/test_service.py b/tests/test_service.py\n"
            "--- a/tests/test_service.py\n"
            "+++ b/tests/test_service.py\n"
            "@@ -60,2 +60,3 @@\n"
            "+    assert backend.get_value() == 3\n"
            "     pass\n"
        )

        examples, _ = build_train_gold_sft_dataset((row,))
        payload = json.loads(examples[0]["messages"][1]["content"])

        # A test that calls no endpoint still asserts something the failure names, so the row
        # keeps the shape instead of falling back to the bare test name.
        self.assertIn("[failing statement] tests/test_service.py:60:", payload["initial_observation"])

    def test_an_observation_that_quotes_the_fix_is_refused(self) -> None:
        # The gold patch adds `return max(0, value)`; a failure output that printed it would hand
        # the policy the answer instead of the evidence, so the builder refuses the row.
        observation = (
            "Baseline verifier result:\nTests failed (exit=1).\n"
            "[last error] AssertionError: assert 0 == 1\n"
            "    return max(0, value)\n"
        )

        with self.assertRaisesRegex(SFTDatasetError, "leaks a gold patch line"):
            _require_no_gold_leak("getmoto__moto-7509", observation, SOURCE_AND_TEST_PATCH)

    def test_an_observation_with_failure_evidence_is_not_a_leak(self) -> None:
        observation = (
            "Baseline verifier result:\nTests failed (exit=1).\n"
            "[failing statement] tests/test_service.py:29: assert calculate(1) == 0\n"
            "[last error] AssertionError: assert 1 == 0\n"
        )

        # Raises nothing: the assertion and the exception are the failure, and the dataset is
        # answer-bearing by design - only the prompt has to stay free of the fix.
        _require_no_gold_leak("getmoto__moto-7509", observation, SOURCE_AND_TEST_PATCH)

    def test_harvested_runtime_lines_are_carried_into_the_observation(self) -> None:
        row = _row()
        harvested = {
            row["instance_id"]: (
                "[last error] AssertionError: assert Decimal('11.7') == Decimal('11.70')",
                "[string values in the failing frame] table_name = 't911877'",
            )
        }

        examples, _ = build_train_gold_sft_dataset((row,), harvested_failures=harvested)
        payload = json.loads(examples[0]["messages"][1]["content"])

        # `[last error]` and the frame's string values are what only a run produces, so a row
        # cannot compose them; they come from the archived verifier output.
        self.assertIn("[last error] AssertionError", payload["initial_observation"])
        self.assertIn("[string values in the failing frame]", payload["initial_observation"])

    def test_a_row_without_harvested_lines_keeps_only_what_the_test_patch_states(self) -> None:
        # This fixture's test patch renames a test and adds no assertion, so the row has no
        # statement to carry and must not invent one. Every pinned row does have one.
        examples, _ = build_train_gold_sft_dataset((_row(),))
        payload = json.loads(examples[0]["messages"][1]["content"])

        self.assertNotIn("[last error]", payload["initial_observation"])
        self.assertNotIn("[string values in the failing frame]", payload["initial_observation"])
        self.assertIn("test_new", payload["initial_observation"])

    def test_locate_prefers_a_runtime_value_over_the_assertion(self) -> None:
        row = _row()
        harvested = {
            row["instance_id"]: (
                "[last error] botocore.exceptions.ClientError: An error occurred "
                "(InvalidServiceName) when calling the DescribeVpcEndpointServices operation",
            )
        }

        examples, _ = build_train_gold_sft_dataset((row,), harvested_failures=harvested)
        locate = next(example for example in examples if example["stage"] == "locate")

        # The runtime line names what the implementation answers, which is the literal the held-out
        # task needs and the assertion does not carry. The service exception's own name wins: it is
        # the most specific name in the line and it appears in the code that raises it.
        self.assertEqual(
            locate["target_action"]["arguments"]["query"],
            "InvalidServiceName",
        )

    def test_locate_teaches_the_failure_literal_not_the_gold_path(self) -> None:
        row = _row()
        row["test_patch"] = (
            "diff --git a/tests/test_core/test_config.py b/tests/test_core/test_config.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/tests/test_core/test_config.py\n"
            "@@ -0,0 +1,3 @@\n"
            "+def test_change_configuration_using_api() -> None:\n"
            "+    resp = requests.get(\"http://motoapi.amazonaws.com/moto-api/config\")\n"
            '+    assert resp.json()["batch"] == {"use_docker": True}\n'
        )

        examples, _ = build_train_gold_sft_dataset((row,))
        locate = next(example for example in examples if example["stage"] == "locate")

        # A locate target naming the gold path is a query the policy cannot derive and one the
        # search already answers; the prompt asks for a literal the failure names. The action is
        # what a warm start copies, so the observation above it did not change this on its own.
        self.assertEqual(
            locate["target_action"],
            {"kind": "search_text", "arguments": {"query": "use_docker"}},
        )

    def test_locate_never_teaches_a_test_name_or_a_header_word(self) -> None:
        row = _row()

        examples, _ = build_train_gold_sft_dataset((row,))
        locate = next(example for example in examples if example["stage"] == "locate")

        query = locate["target_action"]["arguments"]["query"]
        self.assertNotIn(query, {"Baseline", "Tests", "assert"})
        self.assertFalse(query.startswith("test_"))

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
