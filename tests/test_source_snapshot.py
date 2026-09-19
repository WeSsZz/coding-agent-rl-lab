from __future__ import annotations

import io
import subprocess
import sys
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "source_snapshot.py"
REQUIRED = ("--require", "pyproject.toml", "src", "tests")


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(("git", *arguments), cwd=repository, check=True, capture_output=True)


def _commit_repository(repository: Path) -> None:
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "snapshot@example.invalid")
    _git(repository, "config", "user.name", "Snapshot")
    _git(repository, "config", "commit.gpgsign", "false")
    (repository / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    (repository / "src").mkdir()
    (repository / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "tests").mkdir()
    (repository / "tests" / "test_app.py").write_text(
        "def test_value():\n    assert True\n", encoding="utf-8"
    )
    _git(repository, "add", "--all")
    _git(repository, "commit", "--quiet", "-m", "initial")


def _members(bundle: Path) -> list[tuple[tarfile.TarInfo, bytes]]:
    with tarfile.open(bundle) as handle:
        return [
            (member, handle.extractfile(member).read())
            for member in handle.getmembers()
            if member.isfile()
        ]


def _rewrite(bundle: Path, replaced: str | None = None, added: str | None = None) -> None:
    """Rebuild the archive with one file's content changed, or one extra file added."""

    entries = _members(bundle)
    with tarfile.open(bundle, "w") as handle:
        for member, data in entries:
            if member.name == replaced:
                data = b"VALUE = 2\n"
            member.size = len(data)
            handle.addfile(member, io.BytesIO(data))
        if added is not None:
            payload = b"extra\n"
            info = tarfile.TarInfo(added)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))


def _unpack(bundle: Path, root: Path) -> None:
    for member, data in _members(bundle):
        target = root / member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


class SourceSnapshotTests(unittest.TestCase):
    def run_script(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            (sys.executable, str(SCRIPT), *arguments),
            capture_output=True,
            text=True,
            check=False,
        )

    def create(
        self, repository: Path, bundle: Path, *extra: str
    ) -> subprocess.CompletedProcess:
        """Create runs from its own repository by default, so every test names one."""

        return self.run_script(
            "create",
            "--repository",
            str(repository),
            "--out",
            str(bundle),
            *extra,
        )

    def test_create_lists_every_tracked_file_and_verifies(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            _commit_repository(repository)
            bundle = repository / "snapshot.tar"

            created = self.create(repository, bundle, *REQUIRED)
            manifest = Path(f"{bundle}.manifest.txt")
            verified = self.run_script("verify", "--bundle", str(bundle), *REQUIRED)
            listed = [
                line.partition("  ")[2]
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(created.returncode, 0, created.stderr)
        self.assertIn("files=3", created.stdout)
        self.assertEqual(listed, ["pyproject.toml", "src/app.py", "tests/test_app.py"])
        self.assertEqual(verified.returncode, 0, verified.stdout)
        self.assertIn("verified files=3", verified.stdout)

    def test_verify_accepts_an_unpacked_tree(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory, "repository")
            repository.mkdir()
            _commit_repository(repository)
            bundle = repository / "snapshot.tar"
            self.create(repository, bundle, *REQUIRED)
            unpacked = Path(directory, "unpacked")
            _unpack(bundle, unpacked)
            manifest = Path(f"{bundle}.manifest.txt")

            clean = self.run_script(
                "verify", "--tree", str(unpacked), "--manifest", str(manifest), *REQUIRED
            )
            (unpacked / "src" / "app.py").write_text("VALUE = 3\n", encoding="utf-8")
            changed = self.run_script(
                "verify", "--tree", str(unpacked), "--manifest", str(manifest), *REQUIRED
            )

        self.assertEqual(clean.returncode, 0, clean.stdout)
        self.assertEqual(changed.returncode, 1)
        self.assertIn("sha256 mismatch: src/app.py", changed.stdout)

    def test_verify_reports_a_tampered_archive(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            _commit_repository(repository)
            bundle = repository / "snapshot.tar"
            self.create(repository, bundle, *REQUIRED)
            _rewrite(bundle, replaced="src/app.py")

            verified = self.run_script("verify", "--bundle", str(bundle), *REQUIRED)

        self.assertEqual(verified.returncode, 1)
        self.assertIn("sha256 mismatch: src/app.py", verified.stdout)
        self.assertIn("FAILED", verified.stdout)

    def test_verify_names_a_bundle_that_is_not_there(self) -> None:
        with TemporaryDirectory() as directory:
            missing = Path(directory, "absent.tar")

            verified = self.run_script("verify", "--bundle", str(missing), *REQUIRED)

        self.assertEqual(verified.returncode, 1)
        self.assertIn(f"no such bundle: {missing}", verified.stderr)

    def test_verify_reports_a_file_the_manifest_does_not_list(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            _commit_repository(repository)
            bundle = repository / "snapshot.tar"
            self.create(repository, bundle, *REQUIRED)
            _rewrite(bundle, added="src/extra.py")

            verified = self.run_script("verify", "--bundle", str(bundle), *REQUIRED)

        self.assertEqual(verified.returncode, 1)
        self.assertIn("not listed in the manifest: src/extra.py", verified.stdout)

    def test_create_refuses_a_snapshot_that_is_missing_a_required_path(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            _commit_repository(repository)

            created = self.create(
                repository,
                repository / "snapshot.tar",
                "--require",
                "pyproject.toml",
                "datasets",
            )

        self.assertNotEqual(created.returncode, 0)
        self.assertIn("the snapshot is incomplete, missing: datasets", created.stderr)

    def test_create_refuses_a_dirty_working_tree(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            _commit_repository(repository)
            bundle = repository / "snapshot.tar"
            (repository / "src" / "app.py").write_text("VALUE = 4\n", encoding="utf-8")

            created = self.create(repository, bundle, *REQUIRED)
            allowed = self.create(repository, bundle, "--allow-dirty", *REQUIRED)

        self.assertNotEqual(created.returncode, 0)
        self.assertIn("refusing to snapshot a dirty working tree", created.stderr)
        self.assertEqual(allowed.returncode, 0, allowed.stderr)

    def test_create_refuses_to_overwrite_an_existing_snapshot(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            _commit_repository(repository)
            bundle = repository / "snapshot.tar"
            self.create(repository, bundle, *REQUIRED)

            again = self.create(repository, bundle, *REQUIRED)
            forced = self.create(repository, bundle, "--force", *REQUIRED)

        self.assertIn("refusing to overwrite", again.stderr)
        self.assertEqual(forced.returncode, 0, forced.stderr)
