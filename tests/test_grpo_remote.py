from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from coding_agent_rl_lab.evaluation import load_builtin_tasks
from coding_agent_rl_lab.contracts import ActionKind, AgentAction
from coding_agent_rl_lab.grpo_remote import (
    GRPOWorker,
    RemoteGRPOCodingEnvironment,
    add_navigation_evidence,
    add_parent_path_evidence,
    build_parser,
    build_worker_server,
)
from coding_agent_rl_lab.providers import LocalFixtureEnvironmentProvider


class GRPORemoteTests(unittest.TestCase):
    def test_navigation_evidence_extracts_failure_object_before_exception(self):
        observation = (
            "Exception=FailureEventException at '(StateTaskServiceAwsSdk| {'details': {}}'"
        )

        annotated = add_navigation_evidence(observation)

        self.assertIn("OBJECT_UNDER_FAILURE:StateTaskServiceAwsSdk", annotated)
        self.assertIn("EXCEPTION_CLASS:FailureEventException", annotated)
        self.assertLess(annotated.index("OBJECT_UNDER_FAILURE"), annotated.index("EXCEPTION_CLASS"))

    def test_parent_path_evidence_resolves_local_imported_base(self):
        observation = """from external.base import ExternalBase
from moto.pkg.callback import (
    CallbackBase,
)

class Service(CallbackBase):
    pass
"""

        annotated = add_parent_path_evidence(observation, "moto/pkg/service.py")

        self.assertIn("PARENT_IMPLEMENTATION_PATH:moto/pkg/callback.py", annotated)
        self.assertNotIn("external/base.py", annotated)

    def test_navigation_only_disables_post_reset_verifier(self):
        root = Path(__file__).resolve().parents[1]
        tasks = load_builtin_tasks(root)
        worker = GRPOWorker(
            {task.task_id: task for task in tasks}, LocalFixtureEnvironmentProvider(root),
            navigation_only=True,
        )
        try:
            session = worker.create("clamp-negative-values")["session_id"]
            run = worker.action(session, AgentAction(ActionKind.RUN_TESTS))
            self.assertFalse(run["terminated"])
            self.assertIn("disabled", run["observation"])
            finish = worker.action(session, AgentAction(ActionKind.FINISH))
            self.assertTrue(finish["terminated"])
            self.assertEqual(finish["reward"], 0.0)
        finally:
            worker.close()

    def test_worker_records_explicit_conservative_reward_version(self):
        args = build_parser().parse_args(["--reward-version", "conservative-v2"])
        root = Path(__file__).resolve().parents[1]
        tasks = load_builtin_tasks(root)
        worker = GRPOWorker({task.task_id: task for task in tasks},
                            LocalFixtureEnvironmentProvider(root), reward_version=args.reward_version)
        try:
            session = worker.create("clamp-negative-values")
            payload = worker.finalize(session["session_id"])
            self.assertEqual(payload["reward_components"]["reward_version"], "conservative-v2")
            self.assertEqual(payload["reward"], 0.0)
        finally:
            worker.close()

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

    def test_remote_environment_can_replace_a_previously_read_line_range(self) -> None:
        environment = RemoteGRPOCodingEnvironment(self.base_url, self.token)
        environment.reset(task_id="clamp-negative-values")
        environment.read_file("values.py")
        updated = environment.replace_lines(
            "values.py",
            4,
            4,
            "    return max(0, value)",
        )

        self.assertEqual(updated, "Updated values.py.")
        self.assertIn("OK", environment.finish())
        self.assertEqual(environment.reward, 1.0)

    def test_remote_environment_returns_small_reward_for_verified_valid_patch(self) -> None:
        environment = RemoteGRPOCodingEnvironment(self.base_url, self.token)
        environment.reset(task_id="clamp-negative-values")
        environment.read_file("values.py")
        environment.replace_text(
            "values.py",
            "    return value\n",
            "    return max(1, value)\n",
        )
        self.assertIn("FAILED", environment.finish())

        self.assertEqual(environment.reward, 0.15)
        self.assertEqual(environment.get_reward(), 0.15)

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

    def test_remote_client_writes_answer_free_reward_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "reward-audit.jsonl"
            environment = RemoteGRPOCodingEnvironment(
                self.base_url,
                self.token,
                reward_audit_path=audit_path,
            )
            environment.reset(task_id="clamp-negative-values")
            ranged = environment.read_file("values.py", start_line=1, end_line=20)
            self.assertIn("clamp_non_negative", ranged)
            # `finish` is refused while the verifier fails and no source edit exists, so
            # the audited action completion now has to carry the edit with it.
            environment.replace_text(
                "values.py",
                "    return value\n",
                "    return max(0, value)\n",
            )
            environment.finish()

            records = [
                json.loads(line)
                for line in audit_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["task_id"], "clamp-negative-values")
            self.assertEqual(records[0]["completion_source"], "action")
            self.assertIn("strict_reward", records[0])
            self.assertIn("reward_components", records[0])
            self.assertEqual(
                records[0]["action_kinds"],
                ["read_file", "replace_text", "finish"],
            )
            self.assertEqual(
                records[0]["action_outcomes"],
                [
                    {"kind": "read_file", "outcome": "ok"},
                    {"kind": "replace_text", "outcome": "updated"},
                    {"kind": "finish", "outcome": "terminated"},
                ],
            )
            self.assertNotIn("observation", records[0])
            self.assertNotIn("token", records[0])

    def test_unpatched_finish_is_refused_over_the_wire_then_finalize_scores_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "reward-audit.jsonl"
            environment = RemoteGRPOCodingEnvironment(
                self.base_url,
                self.token,
                reward_audit_path=audit_path,
            )
            environment.reset(task_id="clamp-negative-values")
            environment.read_file("values.py")
            refused = environment.finish()

            self.assertIn("finish refused", refused)
            self.assertEqual(environment.reward, 0.0)
            records = [
                json.loads(line)
                for line in audit_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["completion_source"], "finalize")
            self.assertEqual(records[0]["action_kinds"], ["read_file", "finish"])

    def test_inactive_probe_reward_is_safe_for_trl_introspection(self) -> None:
        environment = RemoteGRPOCodingEnvironment(self.base_url, self.token)

        self.assertEqual(environment.reward, 0.0)
        self.assertEqual(environment.get_reward(), 0.0)


if __name__ == "__main__":
    unittest.main()
