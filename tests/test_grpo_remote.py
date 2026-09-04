from __future__ import annotations

import threading
import unittest
from pathlib import Path

from coding_agent_rl_lab.evaluation import load_builtin_tasks
from coding_agent_rl_lab.grpo_remote import (
    GRPOWorker,
    RemoteGRPOCodingEnvironment,
    build_parser,
    build_worker_server,
)
from coding_agent_rl_lab.providers import LocalFixtureEnvironmentProvider


class GRPORemoteTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1]
        tasks = load_builtin_tasks(root)
        worker = GRPOWorker(
            {task.task_id: task for task in tasks},
            LocalFixtureEnvironmentProvider(root),
        )
        self.token = "x" * 32
        self.server = build_worker_server(worker, token=self.token, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_remote_environment_completes_successful_fixture(self) -> None:
        environment = RemoteGRPOCodingEnvironment(self.base_url, self.token)
        initial = environment.reset(task_id="clamp-negative-values")

        self.assertIn("Tests failed", initial)
        environment.read_file("values.py")
        environment.replace_text(
            "values.py",
            "    return value\n",
            "    return max(0, value)\n",
        )
        self.assertIn("OK", environment.finish())
        self.assertEqual(environment.reward, 1.0)
        self.assertEqual(environment.get_reward(), 1.0)

    def test_invalid_token_is_rejected(self) -> None:
        environment = RemoteGRPOCodingEnvironment(self.base_url, "y" * 32)

        with self.assertRaisesRegex(Exception, "worker rejected request"):
            environment.reset(task_id="clamp-negative-values")

    def test_worker_cli_supports_fixture_curriculum(self) -> None:
        args = build_parser().parse_args(
            ["--task-source", "fixtures", "--task-count", "2"]
        )

        self.assertEqual(args.task_source, "fixtures")
        self.assertEqual(args.task_count, 2)

    def test_worker_cli_supports_explicit_swe_gym_task_set(self) -> None:
        args = build_parser().parse_args(
            ["--task-source", "swe-gym", "--task-set", "regression", "--task-count", "2"]
        )

        self.assertEqual(args.task_set, "regression")
        self.assertEqual(args.task_count, 2)

    def test_worker_cli_supports_exact_swe_gym_task_id(self) -> None:
        args = build_parser().parse_args(
            ["--task-source", "swe-gym", "--task-set", "train", "--task-id", "getmoto__moto-7509"]
        )

        self.assertEqual(args.task_id, ["getmoto__moto-7509"])
        self.assertIsNone(args.task_count)

    def test_worker_cli_supports_bounded_test_timeout(self) -> None:
        args = build_parser().parse_args(["--test-timeout-seconds", "180"])

        self.assertEqual(args.test_timeout_seconds, 180.0)

    def test_inactive_probe_reward_is_safe_for_trl_introspection(self) -> None:
        environment = RemoteGRPOCodingEnvironment(self.base_url, self.token)

        self.assertEqual(environment.reward, 0.0)
        self.assertEqual(environment.get_reward(), 0.0)


if __name__ == "__main__":
    unittest.main()
