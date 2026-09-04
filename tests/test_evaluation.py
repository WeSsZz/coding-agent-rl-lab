from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_agent_rl_lab.contracts import ActionKind, AgentAction, PolicyDecision, PolicyManifest, Trajectory
from coding_agent_rl_lab.evaluation import evaluate
from coding_agent_rl_lab.evaluation import load_builtin_tasks
from coding_agent_rl_lab.providers import LocalFixtureEnvironmentProvider
from coding_agent_rl_lab.rollout import RolloutCollector
from coding_agent_rl_lab.rollout import write_report, write_trajectories


class ProtocolFailurePolicy:
    manifest = PolicyManifest(
        policy_id="protocol-failure",
        version="1",
        policy_type="test",
    )

    def next_action(self, task, history, *, seed=None, initial_observation=""):
        del task, history, initial_observation
        return PolicyDecision(
            action=AgentAction(ActionKind.FINISH),
            input_messages=({"role": "user", "content": "test"},),
            output_text="invalid output",
            metadata={"seed": seed, "usage": {"total_tokens": 2}},
            violation="policy_protocol_error",
        )


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_noop_is_an_expected_failure_control(self) -> None:
        report, trajectories = evaluate(self.root, policy_name="noop", repetitions=1)
        self.assertEqual(report["pass_at_1"], 0.0)
        self.assertFalse(report["training_performed"])
        self.assertTrue(all(not item.reward.task_success for item in trajectories))

    def test_reference_pipeline_is_reliable_across_three_trials(self) -> None:
        report, trajectories = evaluate(self.root, policy_name="reference", repetitions=3)
        self.assertEqual(report["trial_count"], 6)
        self.assertEqual(report["pass_at_1"], 1.0)
        self.assertEqual(report["pass_power_3"], 1.0)
        self.assertEqual(report["fully_reliable_task_rate"], 1.0)
        self.assertEqual(report["success_sample_variance"], 0.0)
        self.assertIsNotNone(report["scalar_reward_sample_variance"])
        self.assertEqual(
            report["task_reliability"]["off-by-one-inclusive-range"]["success_sample_variance"],
            0.0,
        )
        self.assertEqual(report["policy"]["metadata"]["contains_answers"], True)
        self.assertTrue(all(item.changed_files for item in trajectories))

    def test_repeated_collection_uses_distinct_repetition_numbers_and_seed_ranges(self) -> None:
        task = load_builtin_tasks(self.root)[0]
        collector = RolloutCollector(LocalFixtureEnvironmentProvider(self.root))
        sentinel_trajectories = (object(), object(), object())
        with patch.object(collector, "collect", side_effect=sentinel_trajectories) as collect:
            trajectories = collector.collect_repetitions(
                task,
                ProtocolFailurePolicy(),
                repetitions=3,
                base_seed=12345,
            )

        self.assertEqual(trajectories, sentinel_trajectories)
        self.assertEqual(
            [(call.kwargs["repetition"], call.kwargs["seed"]) for call in collect.call_args_list],
            [(1, 12345), (2, 22345), (3, 32345)],
        )

    def test_repeated_collection_rejects_invalid_counts_and_overlapping_seed_ranges(self) -> None:
        task = load_builtin_tasks(self.root)[0]
        collector = RolloutCollector(LocalFixtureEnvironmentProvider(self.root))

        with self.assertRaisesRegex(ValueError, "repetitions"):
            collector.collect_repetitions(
                task,
                ProtocolFailurePolicy(),
                repetitions=0,
                base_seed=1,
            )
        with self.assertRaisesRegex(ValueError, "seed_stride"):
            collector.collect_repetitions(
                task,
                ProtocolFailurePolicy(),
                repetitions=2,
                base_seed=1,
                seed_stride=task.max_steps - 1,
            )

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
        self.assertEqual(loaded["success_sample_variance"], 0.0)
        self.assertIsNone(
            loaded["task_reliability"]["off-by-one-inclusive-range"]["success_sample_variance"]
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["schema_version"], 3)
        self.assertIn("Baseline verifier result", rows[0]["initial_observation"])
        self.assertEqual(Trajectory.from_dict(rows[0]), trajectories[0])

    def test_policy_io_and_protocol_failure_are_preserved_in_trajectory(self) -> None:
        task = load_builtin_tasks(self.root)[0]
        trajectory = RolloutCollector(LocalFixtureEnvironmentProvider(self.root)).collect(
            task,
            ProtocolFailurePolicy(),
            repetition=1,
            seed=77,
        )

        self.assertEqual(trajectory.steps[0].policy_output, "invalid output")
        self.assertEqual(trajectory.steps[0].policy_input[0]["role"], "user")
        self.assertEqual(trajectory.steps[0].policy_metadata["seed"], 77)
        self.assertIn("policy_protocol_error", trajectory.reward.violations)
        self.assertEqual(trajectory.reward.scalar, 0.0)


if __name__ == "__main__":
    unittest.main()
