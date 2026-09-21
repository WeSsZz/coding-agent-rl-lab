from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_swe_gym_test_command import (  # noqa: E402
    build_command,
    parse_prefixes,
    prefix_for,
)


class ParsePrefixesTests(unittest.TestCase):
    def test_a_bare_entry_becomes_the_default(self) -> None:
        scoped, default = parse_prefixes(["/opt/pytest -rA"])

        self.assertEqual(scoped, {})
        self.assertEqual(default, ("/opt/pytest", "-rA"))

    def test_a_scoped_entry_is_keyed_by_repo_and_version(self) -> None:
        # The scope is `<repo>/<version>=<command>`, and the repo itself contains a slash, so the
        # split has to happen on the last slash.
        scoped, default = parse_prefixes(["conan-io/conan/2.0=/opt/pytest -n0"])

        self.assertIsNone(default)
        self.assertEqual(scoped[("conan-io/conan", "2.0")], ("/opt/pytest", "-n0"))

    def test_a_misspelled_scope_fails_loudly_at_selection_time(self) -> None:
        # `conan-io/conan=2.0=...` would key on ("conan-io", "conan"), which matches no row.
        scoped, default = parse_prefixes(["conan-io/conan=2.0=/opt/pytest"])

        self.assertIsNone(default)
        with self.assertRaises(ValueError):
            prefix_for({"repo": "conan-io/conan", "version": "2.0"}, scoped, default)

    def test_two_repositories_can_carry_different_prefixes(self) -> None:
        # The real case: conan's image has xdist so `-n0` is fine, pydantic's does not.
        scoped, default = parse_prefixes(
            [
                "conan-io/conan/2.0=/opt/pytest -n0 -rA",
                "pydantic/pydantic/2.7=/opt/pytest -rA -p no:pretty",
            ]
        )

        self.assertIsNone(default)
        self.assertEqual(scoped[("conan-io/conan", "2.0")], ("/opt/pytest", "-n0", "-rA"))
        self.assertEqual(
            scoped[("pydantic/pydantic", "2.7")],
            ("/opt/pytest", "-rA", "-p", "no:pretty"),
        )

    def test_two_unscoped_entries_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            parse_prefixes(["/a", "/b"])

    def test_no_entries_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            parse_prefixes([])


class PrefixSelectionTests(unittest.TestCase):
    def test_a_scoped_prefix_wins_over_the_default(self) -> None:
        scoped, default = parse_prefixes(["/default", "acme/widget/1.0=/scoped"])

        chosen = prefix_for({"repo": "acme/widget", "version": "1.0"}, scoped, default)

        self.assertEqual(chosen, ("/scoped",))

    def test_an_unmatched_repository_falls_back_to_the_default(self) -> None:
        scoped, default = parse_prefixes(["/default", "acme/widget/1.0=/scoped"])

        chosen = prefix_for({"repo": "other/thing", "version": "9"}, scoped, default)

        self.assertEqual(chosen, ("/default",))

    def test_no_default_and_no_match_is_an_error_rather_than_a_silent_skip(self) -> None:
        scoped, default = parse_prefixes(["acme/widget/1.0=/scoped"])

        with self.assertRaises(ValueError):
            prefix_for({"repo": "other/thing", "version": "9"}, scoped, default)


class BuildCommandTests(unittest.TestCase):
    def test_targets_are_deduplicated_and_ordered_fail_to_pass_first(self) -> None:
        command = build_command(
            ("/opt/miniconda3/envs/testbed/bin/pytest", "-n0", "-rA"),
            ("t::a", "t::b"),
            ("t::b", "t::c"),
        )

        self.assertEqual(
            command,
            (
                "/opt/miniconda3/envs/testbed/bin/pytest",
                "-n0",
                "-rA",
                "--",
                "t::a",
                "t::b",
                "t::c",
            ),
        )

    def test_an_empty_pass_to_pass_list_is_allowed(self) -> None:
        command = build_command(("pytest",), ("t::a",), ())

        self.assertEqual(command, ("pytest", "--", "t::a"))

    def test_the_separator_is_present_even_with_no_targets(self) -> None:
        # The adapter's graded-target check slices at "--", so it must always be there.
        self.assertEqual(build_command(("pytest", "-q"), (), ()), ("pytest", "-q", "--"))


class ExportTaskRowsTests(unittest.TestCase):
    """The exporter is the only path by which an answer-bearing row reaches the worker."""

    def test_it_refuses_to_write_outside_work_private(self) -> None:
        from scripts.export_task_rows import main

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rows.jsonl"
            code = main(
                [
                    "--instance-id",
                    "getmoto__moto-7509",
                    "--output",
                    str(output),
                ]
            )

        self.assertEqual(code, 1)
        self.assertFalse(output.exists())

    def test_a_missing_instance_is_reported_rather_than_skipped(self) -> None:
        from scripts.export_task_rows import main

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rows.jsonl"
            code = main(
                [
                    "--instance-id",
                    "getmoto__moto-0000",
                    "--output",
                    str(output),
                    "--allow-tracked",
                ]
            )

        self.assertEqual(code, 1)
        self.assertFalse(output.exists())


class ExportedRowShapeTests(unittest.TestCase):
    """Guard the field names the audit reads, using the real cached export if it exists."""

    CANDIDATES = (
        Path("work/private/swe-gym-train-rows.jsonl"),
        Path("work/private/audit-rows.jsonl"),
    )

    def test_the_cached_export_carries_every_field_the_audit_reads(self) -> None:
        existing = [path for path in self.CANDIDATES if path.is_file()]
        if not existing:
            self.skipTest("no cached row export on this machine")

        required = {"instance_id", "repo", "version", "base_commit", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"}
        for path in existing:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = required - set(row)
                self.assertFalse(missing, f"{path}: missing {sorted(missing)}")
                self.assertTrue(row["base_commit"], f"{path}: empty base_commit")
                self.assertIsInstance(row["FAIL_TO_PASS"], list)


if __name__ == "__main__":
    unittest.main()
