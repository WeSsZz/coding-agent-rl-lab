from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from coding_agent_rl_lab.evaluation import load_builtin_tasks
from coding_agent_rl_lab.contracts import DatasetSplit, PolicyManifest
from coding_agent_rl_lab.swe_gym_rollout import (
    build_parser,
    collect_trajectories_incrementally,
    write_rollout_checkpoint,
)


class SWEGymRolloutCommandTests(unittest.TestCase):
    def test_repetitions_defaults_to_one(self) -> None:
        args = build_parser().parse_args(["--model", "example/coder"])

        self.assertEqual(args.repetitions, 1)

    def test_repetitions_and_base_seed_are_configurable(self) -> None:
        args = build_parser().parse_args(
            [
                "--model",
                "example/coder",
                "--repetitions",
                "5",
                "--seed",
                "9876",
            ]
        )

        self.assertEqual(args.repetitions, 5)
        self.assertEqual(args.seed, 9876)

    def test_task_count_is_configurable(self) -> None:
        args = build_parser().parse_args(
            ["--model", "example/coder", "--task-count", "10"]
        )

        self.assertEqual(args.task_count, 10)

    def test_task_set_is_configurable(self) -> None:
        args = build_parser().parse_args(
            ["--model", "example/coder", "--task-set", "held-out", "--task-count", "2"]
        )

        self.assertEqual(args.task_set, "held-out")
        self.assertEqual(args.task_count, 2)

    def test_exact_task_id_is_configurable(self) -> None:
        args = build_parser().parse_args(
            ["--model", "example/coder", "--task-set", "train", "--task-id", "getmoto__moto-7509"]
        )

        self.assertEqual(args.task_id, ["getmoto__moto-7509"])
        self.assertIsNone(args.task_count)

    def test_test_timeout_is_configurable(self) -> None:
        args = build_parser().parse_args(
            ["--model", "example/coder", "--test-timeout-seconds", "180"]
        )

        self.assertEqual(args.test_timeout_seconds, 180.0)

    def test_resume_is_configurable(self) -> None:
        args = build_parser().parse_args(["--model", "example/coder", "--resume"])

        self.assertTrue(args.resume)

    def test_repetitions_must_be_positive(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                ["--model", "example/coder", "--repetitions", "0"]
            )

    def test_incremental_collection_checkpoints_completed_trajectories_before_failure(self) -> None:
        task = load_builtin_tasks(Path(__file__).resolve().parents[1])[0]
        first = Mock()
        collector = Mock()
        collector.collect.side_effect = [first, RuntimeError("interrupted")]
        progress = []

        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            collect_trajectories_incrementally(
                (task,),
                Mock(),
                collector,
                repetitions=2,
                base_seed=123,
                on_progress=lambda trajectories, planned: progress.append((trajectories, planned)),
            )

        self.assertEqual(progress, [((), 2), ((first,), 2)])
        self.assertEqual(
            [call.kwargs["seed"] for call in collector.collect.call_args_list],
            [123, 10123],
        )

    def test_checkpoint_marks_partial_run_and_writes_empty_trajectory_file(self) -> None:
        task = load_builtin_tasks(Path(__file__).resolve().parents[1])[0]
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            trajectory_path = Path(directory) / "trajectories.jsonl"
            write_rollout_checkpoint(
                (task,),
                (),
                repetitions=2,
                task_set="train",
                split=DatasetSplit.DEVELOPMENT,
                planned_trial_count=2,
                report_path=report_path,
                trajectory_path=trajectory_path,
                run_complete=False,
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertEqual(trajectory_path.read_text(encoding="utf-8"), "")
            self.assertEqual(report["planned_trial_count"], 2)
            self.assertEqual(report["completed_trial_count"], 0)
            self.assertFalse(report["run_complete"])

    def test_resume_skips_compatible_completed_trajectory(self) -> None:
        task = load_builtin_tasks(Path(__file__).resolve().parents[1])[0]
        policy = Mock()
        policy.manifest = PolicyManifest("test-policy", "1", "test")
        existing = Mock(
            task_id=task.task_id,
            repetition=1,
            seed=123,
            policy=policy.manifest,
        )
        second = Mock()
        collector = Mock()
        collector.collect.return_value = second

        trajectories = collect_trajectories_incrementally(
            (task,),
            policy,
            collector,
            repetitions=2,
            base_seed=123,
            existing_trajectories=(existing,),
        )

        self.assertEqual(trajectories, (existing, second))
        collector.collect.assert_called_once_with(
            task,
            policy,
            repetition=2,
            seed=10123,
        )

    def test_resume_rejects_policy_mismatch(self) -> None:
        task = load_builtin_tasks(Path(__file__).resolve().parents[1])[0]
        policy = Mock()
        policy.manifest = PolicyManifest("current", "1", "test")
        existing = Mock(
            task_id=task.task_id,
            repetition=1,
            seed=123,
            policy=PolicyManifest("stale", "1", "test"),
        )

        with self.assertRaisesRegex(ValueError, "policy mismatch"):
            collect_trajectories_incrementally(
                (task,),
                policy,
                Mock(),
                repetitions=1,
                base_seed=123,
                existing_trajectories=(existing,),
            )


if __name__ == "__main__":
    unittest.main()
