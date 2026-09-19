"""Build and verify a complete, checksummed snapshot of this repository.

The repository root once carried a hand-made `codex-sync-v3.tar` that held eight of the
tracked files: enough to look like a snapshot, not enough to run the suites, and nothing
inside it said so. A snapshot is only worth shipping if it is complete and if the machine
that receives it can check that it is, so the file list comes from git rather than from a
glob, and the checksum manifest travels beside the archive.

    python scripts/source_snapshot.py create                 # snapshot-<rev>.tar + manifest
    python scripts/source_snapshot.py verify --bundle FILE   # check an archive
    python scripts/source_snapshot.py verify --tree DIR      # check an unpacked copy

The manifest is plain `sha256sum` output, so a copy that was unpacked on another machine
can be checked there without this tool:

    cd unpacked && sha256sum -c snapshot-<rev>.tar.manifest.txt
"""

from __future__ import annotations

import argparse
import hashlib
import io
import subprocess
import tarfile
from pathlib import Path
from typing import Iterable, Sequence

MANIFEST_SUFFIX = ".manifest.txt"
REPOSITORY = Path(__file__).resolve().parents[1]
# The suites import `scripts.*` and read `datasets` and `fixtures`, so a snapshot without
# them fails as import errors and missing files rather than as anything a reader can
# attribute to a bad upload.
DEFAULT_REQUIRED = ("pyproject.toml", "src", "tests", "scripts", "datasets", "fixtures")


def _git(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        failure = completed.stderr.decode("utf-8", "replace").strip()
        raise SystemExit(f"git {' '.join(arguments)} failed: {failure}")
    return completed.stdout


def _digest(stream: io.BufferedIOBase) -> str:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def _archive_members(archive: bytes) -> list[tuple[str, str]]:
    """The regular files in a tar archive as (path, sha256), sorted by path."""

    members: list[tuple[str, str]] = []
    with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
        for member in handle.getmembers():
            if not member.isfile():
                continue
            stream = handle.extractfile(member)
            if stream is not None:
                members.append((member.name, _digest(stream)))
    return sorted(members)


def _tree_members(root: Path) -> list[tuple[str, str]]:
    members: list[tuple[str, str]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            members.append((path.relative_to(root).as_posix(), _digest(stream)))
    return sorted(members)


def _manifest_text(members: Sequence[tuple[str, str]]) -> str:
    return "".join(f"{digest}  {path}\n" for path, digest in members)


def _read_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        digest, separator, name = line.partition("  ")
        if separator != "  " or len(digest) != 64 or not name:
            raise SystemExit(f"{path}:{number}: expected '<sha256>  <path>'")
        entries[name] = digest
    return entries


def _absent_required(members: Iterable[tuple[str, str]], required: Sequence[str]) -> list[str]:
    names = [path for path, _ in members]
    return [
        item
        for item in required
        if not any(name == item or name.startswith(f"{item}/") for name in names)
    ]


def _dirty_paths(repository: Path, outputs: Sequence[Path]) -> list[str]:
    """Working-tree changes the archive would not contain, excluding this tool's outputs.

    A snapshot written into the repository counts as an untracked file, and refusing the
    next snapshot because of the previous one would be a trap, so the paths passed here
    are filtered out of the report.
    """

    written = {
        path.relative_to(repository).as_posix()
        for path in outputs
        if repository in path.parents
    }
    dirty: list[str] = []
    for line in _git(repository, "status", "--porcelain").decode("utf-8", "replace").splitlines():
        name = line[3:].strip().strip('"').rpartition(" -> ")[2]
        if name and name not in written:
            dirty.append(line)
    return dirty


def create(arguments: argparse.Namespace) -> int:
    repository = arguments.repository.resolve()
    revision = _git(repository, "rev-parse", "--short", arguments.rev)
    short = revision.decode("ascii", "replace").strip()
    bundle = arguments.out or Path(f"snapshot-{short}.tar")
    if not bundle.is_absolute():
        bundle = repository / bundle
    manifest = bundle.with_name(bundle.name + MANIFEST_SUFFIX)
    for path in (bundle, manifest):
        if path.exists() and not arguments.force:
            raise SystemExit(f"refusing to overwrite {path}; pass --force")
    if not arguments.allow_dirty:
        dirty = _dirty_paths(repository, (bundle, manifest))
        if dirty:
            raise SystemExit(
                "refusing to snapshot a dirty working tree: git archive writes the commit, "
                "not these files, so the snapshot would be missing them. Commit or stash "
                "them, or pass --allow-dirty.\n" + "\n".join(dirty)
            )
    archive = _git(repository, "archive", "--format=tar", arguments.rev)
    members = _archive_members(archive)
    absent = _absent_required(members, arguments.require)
    if absent:
        raise SystemExit(f"the snapshot is incomplete, missing: {', '.join(absent)}")
    bundle.write_bytes(archive)

    # Text mode writes `\r\n` for `\n` on Windows, and a checksum checker is free to keep the
    # carriage return as part of the file name - the one on the training host reported every
    # listed file as missing. The manifest is written exactly as `sha256sum` would print it.
    manifest.write_text(_manifest_text(members), encoding="utf-8", newline="\n")
    print(f"rev={short} files={len(members)} bundle={bundle}")
    print(f"manifest={manifest}")
    return 0


def verify(arguments: argparse.Namespace) -> int:
    if arguments.bundle is not None:
        if not arguments.bundle.is_file():
            raise SystemExit(f"no such bundle: {arguments.bundle}")
        members = _archive_members(arguments.bundle.read_bytes())
        default_manifest = arguments.bundle.with_name(
            arguments.bundle.name + MANIFEST_SUFFIX
        )
    else:
        if not arguments.tree.is_dir():
            raise SystemExit(f"no such tree: {arguments.tree}")
        members = _tree_members(arguments.tree)
        default_manifest = None
    manifest_path = arguments.manifest or default_manifest
    if manifest_path is None or not manifest_path.exists():
        raise SystemExit(
            "no manifest to compare against; pass --manifest, or unpack the archive "
            "beside the manifest file it shipped with"
        )
    expected = _read_manifest(manifest_path)
    found = dict(members)
    problems = [
        f"sha256 mismatch: {path}" if path in found else f"missing from the snapshot: {path}"
        for path in sorted(expected)
        if found.get(path) != expected[path]
    ]
    problems.extend(
        f"not listed in the manifest: {path}"
        for path in sorted(found)
        if path not in expected
    )
    problems.extend(
        f"required path is absent: {item}"
        for item in _absent_required(members, arguments.require)
    )
    if problems:
        print("\n".join(problems))
        print(f"FAILED files={len(found)} problems={len(problems)} manifest={manifest_path}")
        return 1
    print(f"verified files={len(found)} manifest={manifest_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    creator = commands.add_parser("create", help="write a snapshot archive and its manifest")
    creator.add_argument("--repository", type=Path, default=REPOSITORY)
    creator.add_argument("--rev", default="HEAD")
    creator.add_argument("--out", type=Path, default=None)
    creator.add_argument("--require", nargs="*", default=list(DEFAULT_REQUIRED))
    creator.add_argument("--force", action="store_true")
    creator.add_argument(
        "--allow-dirty",
        action="store_true",
        help="snapshot the commit even though the working tree has uncommitted changes",
    )
    creator.set_defaults(handler=create)

    verifier = commands.add_parser("verify", help="check an archive or an unpacked copy")
    source = verifier.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path, default=None)
    source.add_argument("--tree", type=Path, default=None)
    verifier.add_argument("--manifest", type=Path, default=None)
    verifier.add_argument("--require", nargs="*", default=list(DEFAULT_REQUIRED))
    verifier.set_defaults(handler=verify)

    arguments = parser.parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
