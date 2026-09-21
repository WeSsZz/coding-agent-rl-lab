from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verify_route_manifest import (  # noqa: E402
    git_blob_hash,
    is_git_worktree,
    sha256_of,
    verify,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _manifest(**overrides: object) -> dict:
    base = {
        "manifest_id": "test",
        "source": {"frozen_commit": "HEAD", "repo_files": []},
        "external_artifacts": [],
    }
    base.update(overrides)
    return base


class VerifyRouteManifestTests(unittest.TestCase):
    @unittest.skipUnless(is_git_worktree(REPOSITORY_ROOT), "needs a git checkout")
    def test_git_blob_hash_matches_the_repository_object(self) -> None:
        expected = subprocess.run(
            ["git", "rev-parse", "HEAD:README.md"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        self.assertEqual(git_blob_hash(REPOSITORY_ROOT, "HEAD", "README.md"), expected)

    @unittest.skipUnless(is_git_worktree(REPOSITORY_ROOT), "needs a git checkout")
    def test_a_wrong_blob_is_reported_as_a_failure(self) -> None:
        manifest = _manifest(
            source={
                "frozen_commit": "HEAD",
                "repo_files": [{"repo_path": "README.md", "git_blob": "0" * 40}],
            }
        )

        lines = verify(manifest, strict=False, repository_root=REPOSITORY_ROOT)

        self.assertTrue(any(line.startswith("  FAIL  README.md") for line in lines))
        self.assertIn("RESULT: FAILED", lines)

    def test_a_tree_without_git_reports_blob_checks_as_unavailable(self) -> None:
        """The VM validation bundle is a tarball with no `.git`; that must not read as a defect."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _manifest(
                source={
                    "frozen_commit": "HEAD",
                    "repo_files": [{"repo_path": "README.md", "git_blob": "0" * 40}],
                }
            )

            lines = verify(manifest, strict=False, repository_root=root)
            self.assertFalse(is_git_worktree(root))

        self.assertTrue(any("unavailable" in line for line in lines), lines)
        self.assertIn("RESULT: OK", lines)

    def test_an_absent_external_artifact_is_only_fatal_in_strict_mode(self) -> None:
        manifest = _manifest(
            external_artifacts=[{"path": "does/not/exist.bin", "sha256": "a" * 64}]
        )

        relaxed = verify(manifest, strict=False, repository_root=REPOSITORY_ROOT)
        strict = verify(manifest, strict=True, repository_root=REPOSITORY_ROOT)

        self.assertTrue(any("absent  does/not/exist.bin" in line for line in relaxed))
        self.assertIn("RESULT: OK", relaxed)
        self.assertIn("RESULT: FAILED", strict)

    def test_a_placeholder_hash_is_refused_rather_than_compared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "artifact.bin").write_bytes(b"payload")
            manifest = _manifest(
                external_artifacts=[
                    {"path": "artifact.bin", "sha256": "UNVERIFIED_PLACEHOLDER"}
                ]
            )

            lines = verify(manifest, strict=False, repository_root=Path(directory))

        self.assertTrue(any("does not record a real sha256" in line for line in lines), lines)
        self.assertIn("RESULT: FAILED", lines)

    def test_a_matching_sha256_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "artifact.bin").write_bytes(b"payload")
            digest = hashlib.sha256(b"payload").hexdigest()
            manifest = _manifest(
                external_artifacts=[{"path": "artifact.bin", "sha256": digest}]
            )

            lines = verify(manifest, strict=True, repository_root=Path(directory))

        self.assertIn("RESULT: OK", lines)

    @unittest.skipUnless(is_git_worktree(REPOSITORY_ROOT), "needs a git checkout")
    def test_the_committed_manifest_verifies(self) -> None:
        manifest = json.loads(
            (REPOSITORY_ROOT / "manifests" / "route-v1-freeze.json").read_text(encoding="utf-8")
        )

        lines = verify(manifest, strict=False, repository_root=REPOSITORY_ROOT)

        self.assertIn("RESULT: OK", lines, "\n".join(lines))

    def test_sha256_of_a_real_file_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "x"
            target.write_bytes(b"abc")

            self.assertEqual(
                sha256_of(target),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )


if __name__ == "__main__":
    unittest.main()
