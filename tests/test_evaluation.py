from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from coding_agent_rl_lab.evaluation import evaluate
from coding_agent_rl_lab.rollout import write_report, write_trajectories


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_noop_is_an_expected_failure_control(self) -> None:
        report, trajectories = evaluate(self.root, policy_name="noop", repetitions=1)
        self.assertEqual(report["pass_at_1"], 0.0)
        self.assertFalse(report["training_performed"])
        self.assertTrue(all(not item.reward.task_success for item in trajectories))
        audit = report["trajectory_dataset"]
        self.assertTrue(audit["integrity_ok"])
        self.assertEqual(audit["training_eligible_count"], 0)
        self.assertEqual(audit["failure_taxonomy"], {"tests_failed_without_patch": 2})

    def test_reference_pipeline_is_reliable_across_three_trials(self) -> None:
        report, trajectories = evaluate(self.root, policy_name="reference", repetitions=3)
        self.assertEqual(report["trial_count"], 6)
        self.assertEqual(report["pass_at_1"], 1.0)
        self.assertEqual(report["pass_power_3"], 1.0)
        self.assertEqual(report["fully_reliable_task_rate"], 1.0)
        self.assertEqual(report["policy"]["metadata"]["contains_answers"], True)
        self.assertTrue(all(item.changed_files for item in trajectories))
        audit = report["trajectory_dataset"]
        self.assertEqual(audit["answer_bearing_count"], 6)
        self.assertEqual(audit["training_eligible_count"], 0)
        self.assertEqual(audit["duplicate_content_groups"], 2)
        self.assertEqual(audit["duplicate_content_records"], 4)

    def test_outputs_are_json_serializable(self) -> None:
        report, trajectories = evaluate(self.root, policy_name="reference", repetitions=1)
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            trajectory_path = Path(directory) / "trajectories.jsonl"
            write_report(report, report_path)
            write_trajectories(trajectories, trajectory_path)
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(loaded["schema_version"], 1)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["schema_version"], 1)
        self.assertEqual(rows[0]["task_split"], "development")
        self.assertEqual(rows[0]["task_provenance"], "repository_owned_fixture")
        self.assertEqual(rows[0]["execution"]["verifier_id"], "local-python-verifier")


if __name__ == "__main__":
    unittest.main()
