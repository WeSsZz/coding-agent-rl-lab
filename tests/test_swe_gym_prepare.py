from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from coding_agent_rl_lab.evaluation import load_builtin_tasks
from coding_agent_rl_lab.swe_gym_prepare import build_parser, prepare_images, write_prompt_rows


class FakeRunner:
    def __init__(self, return_codes: list[int]) -> None:
        self.return_codes = iter(return_codes)
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append(tuple(argv))
        code = next(self.return_codes)
        return subprocess.CompletedProcess(argv, code, "ok" if code == 0 else "", "failed")


class SWEGymPrepareTests(unittest.TestCase):
    def test_task_set_is_configurable(self) -> None:
        args = build_parser().parse_args(["--task-set", "train", "--task-count", "6"])

        self.assertEqual(args.task_set, "train")
        self.assertEqual(args.task_count, 6)

    def test_exact_task_id_is_configurable(self) -> None:
        args = build_parser().parse_args(
            ["--task-set", "held-out", "--task-id", "getmoto__moto-7537"]
        )

        self.assertEqual(args.task_id, ["getmoto__moto-7537"])
        self.assertIsNone(args.task_count)

    def test_prompt_rows_exclude_reference_answers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            count = write_prompt_rows(path, load_builtin_tasks(root))
            text = path.read_text(encoding="utf-8")

        self.assertEqual(count, 2)
        self.assertIn("clamp-negative-values", text)
        self.assertNotIn("reference_actions", text)

    def test_existing_images_are_skipped_and_missing_images_are_pulled(self) -> None:
        runner = FakeRunner([0, 1, 0])

        pulled, skipped = prepare_images(
            ("image:one", "image:two"),
            attempts=3,
            runner=runner,
            sleeper=lambda _: None,
        )

        self.assertEqual((pulled, skipped), (1, 1))
        self.assertEqual(runner.calls[-1], ("docker", "pull", "--quiet", "image:two"))

    def test_failed_pull_is_retried_before_raising(self) -> None:
        runner = FakeRunner([1, 1, 1])

        with self.assertRaisesRegex(RuntimeError, "failed to pull image:one"):
            prepare_images(
                ("image:one",),
                attempts=2,
                runner=runner,
                sleeper=lambda _: None,
            )

        pull_calls = [call for call in runner.calls if call[:2] == ("docker", "pull")]
        self.assertEqual(len(pull_calls), 2)


if __name__ == "__main__":
    unittest.main()
