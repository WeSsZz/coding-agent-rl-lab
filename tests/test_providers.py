from __future__ import annotations

import unittest
from pathlib import Path

from coding_agent_rl_lab.environment import CodingEnvironment, EnvironmentError, LocalFixtureEnvironment
from coding_agent_rl_lab.evaluation import load_builtin_tasks
from coding_agent_rl_lab.providers import (
    DockerSandboxConfig,
    DockerSandboxProvider,
    DockerTaskSpec,
    EnvironmentProvider,
    LocalFixtureEnvironmentProvider,
)


class ProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.task = load_builtin_tasks(self.root)[0]

    def test_local_provider_satisfies_contracts_and_creates_fresh_environments(self) -> None:
        provider = LocalFixtureEnvironmentProvider(self.root)
        first = provider.create(self.task)
        second = provider.create(self.task)
        try:
            self.assertIsInstance(provider, EnvironmentProvider)
            self.assertIsInstance(first, CodingEnvironment)
            self.assertIsInstance(first, LocalFixtureEnvironment)
            self.assertIsNot(first, second)
        finally:
            first.close()
            second.close()

    def test_docker_argv_has_resource_and_security_boundaries(self) -> None:
        argv = DockerSandboxConfig().run_argv("example/task:latest", ("python", "-m", "pytest"))
        self.assertEqual(argv[:2], ("docker", "run"))
        self.assertIn("none", argv)
        self.assertIn("--cap-drop", argv)
        self.assertIn("ALL", argv)
        self.assertIn("no-new-privileges", argv)
        self.assertEqual(argv[-4:], ("example/task:latest", "python", "-m", "pytest"))

    def test_docker_config_rejects_network_access(self) -> None:
        with self.assertRaisesRegex(ValueError, "disable networking"):
            DockerSandboxConfig(network="bridge")

    def test_docker_provider_requires_a_matching_private_spec(self) -> None:
        with self.assertRaisesRegex(EnvironmentError, "no Docker environment spec"):
            DockerSandboxProvider({}).create(self.task)

        spec = DockerTaskSpec(
            task_id=self.task.task_id,
            image="example/task:latest",
            base_commit="different",
            test_patch="--- a/test.py\n+++ b/test.py\n",
            fail_to_pass=("test_bug",),
        )
        with self.assertRaisesRegex(EnvironmentError, "base commit mismatch"):
            DockerSandboxProvider({self.task.task_id: spec}).create(self.task)


if __name__ == "__main__":
    unittest.main()
