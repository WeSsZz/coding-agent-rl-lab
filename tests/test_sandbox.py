from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from coding_agent_rl_lab.sandbox import DockerSandboxProvider, SandboxSpec


class DockerSandboxTests(unittest.TestCase):
    def test_mutable_or_unpinned_images_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with self.assertRaises(ValueError):
                SandboxSpec("python:latest", workspace, ("python", "-V"))
            with self.assertRaises(ValueError):
                SandboxSpec("python", workspace, ("python", "-V"))

    def test_command_has_no_network_and_drops_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = SandboxSpec("python:3.12.2-slim", Path(directory), ("python", "-V"))
            command = DockerSandboxProvider().build_command(spec)
        self.assertIn("none", command)
        self.assertIn("ALL", command)
        self.assertIn("no-new-privileges", command)
        self.assertIn("--read-only", command)
        self.assertEqual(command[-3:], ("python:3.12.2-slim", "python", "-V"))

    def test_secret_like_environment_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                SandboxSpec(
                    "python:3.12.2-slim",
                    Path(directory),
                    ("python", "-V"),
                    environment={"API_KEY": "secret"},
                )

    def test_missing_docker_is_reported_as_unavailable(self) -> None:
        provider = DockerSandboxProvider(docker_executable="definitely-missing-docker")
        self.assertFalse(provider.available())


if __name__ == "__main__":
    unittest.main()
