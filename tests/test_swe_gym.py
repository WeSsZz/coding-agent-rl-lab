from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from coding_agent_rl_lab.swe_gym import (
    SWEGymAdapterError,
    SWEGymInstance,
    SWEPrediction,
    build_swebench_harness_command,
    load_swe_gym_jsonl,
    write_predictions,
)


class SWEGymAdapterTests(unittest.TestCase):
    def test_parses_swe_bench_fields_and_hides_test_patch_from_agent(self) -> None:
        instance = SWEGymInstance.from_dict(
            {
                "instance_id": "owner__repo-1",
                "repo": "owner/repo",
                "base_commit": "abc123",
                "problem_statement": "Fix the bug",
                "FAIL_TO_PASS": '["tests/test_bug.py::test_fix"]',
                "PASS_TO_PASS": ["tests/test_ok.py::test_ok"],
                "test_patch": "secret verifier patch",
            }
        )
        self.assertEqual(instance.fail_to_pass, ("tests/test_bug.py::test_fix",))
        self.assertNotIn("test_patch", instance.agent_input())
        self.assertNotIn("fail_to_pass", instance.agent_input())

    def test_empty_fail_to_pass_is_rejected(self) -> None:
        with self.assertRaises(SWEGymAdapterError):
            SWEGymInstance.from_dict(
                {
                    "instance_id": "x",
                    "repo": "owner/repo",
                    "base_commit": "abc",
                    "problem_statement": "fix",
                    "FAIL_TO_PASS": [],
                }
            )

    def test_jsonl_loader_and_prediction_writer_use_official_shapes(self) -> None:
        row = {
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "base_commit": "abc123",
            "problem_statement": "Fix the bug",
            "FAIL_TO_PASS": ["test_fix"],
            "PASS_TO_PASS": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tasks.jsonl"
            predictions = Path(directory) / "predictions.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            loaded = load_swe_gym_jsonl(source)
            write_predictions((SWEPrediction(loaded[0].instance_id, "model-v1", "diff --git ..."),), predictions)
            saved = json.loads(predictions.read_text(encoding="utf-8"))
        self.assertEqual(saved["instance_id"], "owner__repo-1")
        self.assertEqual(set(saved), {"instance_id", "model_name_or_path", "model_patch"})

    def test_builds_official_harness_command_without_shell(self) -> None:
        command = build_swebench_harness_command(
            dataset_name="SWE-Gym/SWE-Gym",
            predictions_path="work/predictions.jsonl",
            run_id="smoke-1",
            instance_ids=("owner__repo-1",),
        )
        self.assertIn("swebench.harness.run_evaluation", command)
        self.assertIn("--instance_ids", command)
        self.assertNotIn("sh", command)


if __name__ == "__main__":
    unittest.main()
