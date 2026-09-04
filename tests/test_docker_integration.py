from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path

from coding_agent_rl_lab.contracts import ActionKind, AgentAction, CodingTask, DatasetSplit
from coding_agent_rl_lab.docker_environment import (
    DockerSandboxConfig,
    DockerSandboxEnvironment,
    DockerTaskSpec,
    SubprocessCommandRunner,
)


TEST_PATCH = """diff --git a/tests/test_bug.py b/tests/test_bug.py
new file mode 100644
--- /dev/null
+++ b/tests/test_bug.py
@@ -0,0 +1,8 @@
+import unittest
+
+from bug import is_fixed
+
+
+class BugTests(unittest.TestCase):
+    def test_bug_is_fixed(self):
+        self.assertTrue(is_fixed())
"""


@unittest.skipUnless(
    os.environ.get("RUN_DOCKER_INTEGRATION") == "1",
    "set RUN_DOCKER_INTEGRATION=1 to run the real Docker smoke test",
)
class DockerIntegrationTests(unittest.TestCase):
    def test_real_fail_before_patch_pass_after_and_cleanup(self) -> None:
        image = f"coding-agent-rl-lab-smoke:{uuid.uuid4().hex[:12]}"
        container_name: str | None = None

        with tempfile.TemporaryDirectory(prefix="coding-agent-docker-smoke-") as temporary_directory:
            context = Path(temporary_directory)
            repository = context / "repo"
            repository.mkdir()
            (repository / "bug.py").write_text(
                "def is_fixed():\n    return False\n",
                encoding="utf-8",
            )
            self._git(repository, "init")
            self._git(repository, "config", "user.name", "Docker Smoke")
            self._git(repository, "config", "user.email", "docker-smoke@example.invalid")
            self._git(repository, "add", "bug.py")
            self._git(repository, "commit", "-m", "Create failing fixture")
            base_commit = self._git(repository, "rev-parse", "HEAD").stdout.strip()

            (context / "Dockerfile").write_text(
                "FROM python:3.11-slim\n"
                "RUN apt-get update && apt-get install -y --no-install-recommends git "
                "&& rm -rf /var/lib/apt/lists/*\n"
                "WORKDIR /testbed\n"
                "COPY repo/ /testbed/\n",
                encoding="utf-8",
            )

            self._run("docker", "build", "--progress=plain", "--tag", image, str(context))
            task = CodingTask(
                task_id="docker-integration-smoke",
                issue="Make is_fixed return true.",
                fixture_path=None,
                base_commit=base_commit,
                test_command=("python", "-m", "unittest", "discover", "-s", "tests", "-v"),
                split=DatasetSplit.DEVELOPMENT,
                provenance="generated-docker-integration-smoke-v1",
                max_steps=4,
            )
            spec = DockerTaskSpec(
                task_id=task.task_id,
                image=image,
                base_commit=base_commit,
                test_patch=TEST_PATCH,
                fail_to_pass=("tests.test_bug.BugTests.test_bug_is_fixed",),
            )
            environment = DockerSandboxEnvironment(
                spec,
                DockerSandboxConfig(test_timeout_seconds=60.0),
                SubprocessCommandRunner(),
            )

            try:
                observation = environment.reset(task)
                container_name = environment.container_name
                self.assertIn("Baseline tests fail", observation)
                self.assertFalse(environment.baseline_result.passed)

                changed = environment.step(
                    AgentAction(
                        ActionKind.REPLACE_TEXT,
                        {"path": "bug.py", "old": "return False", "new": "return True"},
                    )
                )
                self.assertFalse(changed.terminated)
                tested = environment.step(AgentAction(ActionKind.RUN_TESTS))
                self.assertTrue(tested.terminated)
                self.assertTrue(tested.test_result.passed)
                self.assertEqual(environment.changed_files(), ("bug.py",))
            finally:
                environment.close()
                self._run("docker", "image", "rm", "--force", image)

        self.assertIsNotNone(container_name)
        inspected = subprocess.run(
            ("docker", "inspect", container_name),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(inspected.returncode, 0, "sandbox container was not cleaned up")

    @staticmethod
    def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return DockerIntegrationTests._run("git", "-C", str(repository), *arguments)

    @staticmethod
    def _run(*argv: str) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        if completed.returncode != 0:
            command = " ".join(argv)
            raise AssertionError(
                f"command failed ({completed.returncode}): {command}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        return completed


if __name__ == "__main__":
    unittest.main()
