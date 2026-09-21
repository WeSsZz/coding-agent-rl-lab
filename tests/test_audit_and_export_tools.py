from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_swe_gym_test_command import build_command  # noqa: E402


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
