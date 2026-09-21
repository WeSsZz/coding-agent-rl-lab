"""Recompute and check a route manifest against a working tree.

The point of this script is that a manifest is worthless if its hashes are typed by hand.
Every `repo_files` entry must resolve to the recorded git blob, and every `external_artifacts`
entry that exists on this machine must hash to the recorded sha256. A missing external file is
reported as `absent` rather than as a failure, because `work/` and `remote-artifacts/` are
gitignored and a fresh clone legitimately does not have them.

git blob checks need a real checkout. The VM validation bundle is a plain tarball with no `.git`,
so outside a work tree the blob section is reported as `unavailable` and only fails under
`--strict`.

Usage:
    python scripts/verify_route_manifest.py manifests/route-v1-freeze.json [--strict]

`--strict` turns `absent` and `unavailable` into failures, for use right after an
artifact-producing run in a full checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ManifestError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_git_worktree(repository_root: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def git_blob_hash(repository_root: Path, revision: str, repo_path: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", f"{revision}:{repo_path}"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ManifestError(f"no git object for {revision}:{repo_path}")
    return result.stdout.strip()


def _verify_repo_files(
    manifest: dict, revision: str, repository_root: Path, lines: list[str]
) -> tuple[int, int]:
    """Return (failures, unavailable). Unavailable only becomes fatal under --strict."""

    entries = manifest.get("source", {}).get("repo_files", [])
    if not is_git_worktree(repository_root):
        lines.append("  unavailable  no git work tree here, so blob identity cannot be checked")
        return 0, len(entries)
    failures = 0
    for entry in entries:
        repo_path = entry["repo_path"]
        expected = entry["git_blob"]
        try:
            actual = git_blob_hash(repository_root, revision, repo_path)
        except ManifestError as exc:
            lines.append(f"  FAIL  {repo_path}: {exc}")
            failures += 1
            continue
        if actual == expected:
            lines.append(f"  ok    {repo_path}")
        else:
            lines.append(f"  FAIL  {repo_path}: {expected} != {actual}")
            failures += 1
    return failures, 0


def _verify_external_artifacts(manifest: dict, repository_root: Path, lines: list[str]) -> tuple[int, int]:
    failures = 0
    absent = 0
    for entry in manifest.get("external_artifacts", []):
        relative = entry["path"]
        expected = entry.get("sha256", "")
        if expected.startswith("UNVERIFIED") or len(expected) != 64:
            lines.append(f"  FAIL  {relative}: manifest does not record a real sha256")
            failures += 1
            continue
        target = repository_root / relative
        if not target.is_file():
            lines.append(f"  absent  {relative}")
            absent += 1
            continue
        actual = sha256_of(target)
        if actual == expected:
            lines.append(f"  ok    {relative}")
        else:
            lines.append(f"  FAIL  {relative}: {expected} != {actual}")
            failures += 1
    return failures, absent


def verify(
    manifest: dict, *, strict: bool, repository_root: Path | None = None
) -> list[str]:
    root = Path(repository_root) if repository_root is not None else DEFAULT_REPOSITORY_ROOT
    lines: list[str] = []

    revision = manifest.get("source", {}).get("frozen_commit")
    if not revision:
        raise ManifestError("manifest is missing source.frozen_commit")

    lines.append(f"manifest {manifest.get('manifest_id', '<unnamed>')} @ {revision[:12]}")
    lines.append("")

    lines.append("repo_files (git blob identity):")
    blob_failures, blob_unavailable = _verify_repo_files(manifest, revision, root, lines)

    lines.append("")
    lines.append("external_artifacts (sha256):")
    artifact_failures, absent = _verify_external_artifacts(manifest, root, lines)

    failures = blob_failures + artifact_failures
    if strict:
        failures += absent + blob_unavailable

    lines.append("")
    lines.append(f"absent external artifacts: {absent} (gitignored trees are not in a fresh clone)")
    lines.append(f"RESULT: {'FAILED' if failures else 'OK'}")
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat an absent external artifact or a missing git work tree as a failure",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    lines = verify(manifest, strict=arguments.strict)
    for line in lines:
        print(line)
    return 1 if "RESULT: FAILED" in lines else 0


if __name__ == "__main__":
    sys.exit(main())
