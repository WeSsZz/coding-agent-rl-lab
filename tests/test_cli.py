from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from coding_agent_rl_lab import __main__ as cli


class CLITests(unittest.TestCase):
    def test_swe_plan_validates_instances_and_prints_argv(self) -> None:
        row = {
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "base_commit": "abc123",
            "problem_statement": "Fix the bug",
            "FAIL_TO_PASS": ["test_fix"],
            "PASS_TO_PASS": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "instances.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            argv = [
                "coding-agent-rl",
                "swe-plan",
                "--instances-jsonl",
                str(source),
                "--dataset-name",
                "SWE-Gym/SWE-Gym",
                "--predictions",
                "work/predictions.jsonl",
                "--run-id",
                "smoke-1",
            ]
            output = StringIO()
            with patch("sys.argv", argv), redirect_stdout(output):
                cli.main()
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["validated_instances"], 1)
        self.assertIn("swebench.harness.run_evaluation", payload["command"])

    def test_docker_check_uses_nonzero_exit_when_unavailable(self) -> None:
        argv = ["coding-agent-rl", "docker-check"]
        with patch("sys.argv", argv), patch.object(
            cli.DockerSandboxProvider,
            "available",
            return_value=False,
        ), redirect_stdout(StringIO()):
            with self.assertRaises(SystemExit) as raised:
                cli.main()
        self.assertEqual(raised.exception.code, 1)

    def test_model_policy_requires_endpoint_and_model(self) -> None:
        argv = ["coding-agent-rl", "evaluate", "--policy", "model"]
        with patch("sys.argv", argv):
            with self.assertRaises(SystemExit) as raised:
                cli.main()
        self.assertIn("requires --base-url and --model", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
