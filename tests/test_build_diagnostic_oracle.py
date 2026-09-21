"""Pin the frozen aux-input generator: window rule, real base-commit content, disclosure.

The generator is the only place that decides what conditions B and C show, so these tests run it as
a subprocess on a synthetic task and check the artifacts it writes, rather than trusting a helper.

`test_a_far_apart_second_hunk_is_never_dropped` is the regression test for a real defect: the first
window rule took the union from the first hunk to the last and truncated the tail, so a file whose
hunks sat far apart showed the first region and silently hid the second - on `getmoto__moto-7514`
that hid the whole function the failing tests exercise.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_diagnostic_oracle.py"
SECTION = re.compile(
    r"^----- (?P<path>.+?) \(base commit (?P<commit>[0-9a-f]+), "
    r"lines (?P<spans>[\d,\- ]+?) of (?P<total>\d+)\)$",
    re.M,
)
NOT_SHOWN = re.compile(r"^\[lines (?P<first>\d+)-(?P<last>\d+) not shown", re.M)

TASK_ID = "getmoto__moto-9999"
BASE_COMMIT = "0" * 40
MOD_ADDED = "UNIQUE_ADDED_ALPHA_MARKER"
SECOND_ADDED = "UNIQUE_ADDED_SECOND_MARKER"

GOLD_PATCH = f"""diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,3 +1,4 @@ def alpha():
 context one
+{MOD_ADDED}
 context two
 context three
@@ -40,3 +41,3 @@ def beta():
 context forty
-old value
+changed value here
 context forty two
diff --git a/pkg/second.py b/pkg/second.py
--- a/pkg/second.py
+++ b/pkg/second.py
@@ -1,2 +1,3 @@
+{SECOND_ADDED}
 base one
 base two
"""


def _mod_source() -> str:
    lines = [f"line {number} of the base file" for number in range(1, 61)]
    lines[0] = "context one"
    lines[1] = "context two"
    lines[2] = "context three"
    lines[39] = "context forty"
    lines[40] = "old value"
    lines[41] = "context forty two"
    return "\n".join(lines) + "\n"


def _second_source(include_marker: bool) -> str:
    lines = [f"base {number}" for number in range(1, 11)]
    if include_marker:
        lines[4] = f"    {SECOND_ADDED} already lived here"
    return "\n".join(lines) + "\n"


class BuildDiagnosticOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def build(
        self,
        *,
        include_marker: bool = False,
        pad: int = 20,
        max_lines_per_file: int = 70,
        max_total_lines: int = 200,
        seed_source: bool = True,
        fetch: bool = False,
    ) -> tuple[list[dict], dict, Path]:
        rows = self.root / "rows.jsonl"
        rows.write_text(
            json.dumps(
                {
                    "instance_id": TASK_ID,
                    "base_commit": BASE_COMMIT,
                    "version": "5.0",
                    "test_patch": "diff --git a/tests/test_x.py b/tests/test_x.py\n",
                    "FAIL_TO_PASS": [],
                    "PASS_TO_PASS": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        gold = self.root / "gold.json"
        gold.write_text(json.dumps({TASK_ID: GOLD_PATCH}), encoding="utf-8")
        source_root = self.root / "source-cache"
        if seed_source:
            (source_root / TASK_ID / "pkg").mkdir(parents=True, exist_ok=True)
            (source_root / TASK_ID / "pkg" / "mod.py").write_text(_mod_source(), encoding="utf-8")
            (source_root / TASK_ID / "pkg" / "second.py").write_text(
                _second_source(include_marker), encoding="utf-8"
            )
        output = self.root / "oracle.jsonl"
        manifest = self.root / "oracle-manifest.json"
        arguments = [
            sys.executable,
            str(SCRIPT),
            "--rows-cache",
            str(rows),
            "--gold-patches",
            str(gold),
            "--source-root",
            str(source_root),
            "--task-id",
            TASK_ID,
            "--pad",
            str(pad),
            "--max-lines-per-file",
            str(max_lines_per_file),
            "--max-total-lines",
            str(max_total_lines),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ]
        if not fetch:
            arguments.append("--no-fetch")
        completed = subprocess.run(arguments, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        records = [
            json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line
        ]
        return records, json.loads(manifest.read_text(encoding="utf-8")), source_root

    def by_condition(self, records: list[dict], condition: str) -> dict:
        return next(record for record in records if record["condition"] == condition)

    def window(self, records: list[dict], name: str) -> dict:
        windows = self.by_condition(records, "C")["windows"]
        return next(entry for entry in windows if entry["path"] == f"pkg/{name}")

    def sections(self, message: str) -> list[tuple[re.Match, str]]:
        matches = list(SECTION.finditer(message))
        return [
            (
                match,
                message[match.end() : matches[index + 1].start() if index + 1 < len(matches) else len(message)],
            )
            for index, match in enumerate(matches)
        ]

    def test_condition_a_is_empty_and_b_lists_only_paths(self) -> None:
        records, _, _ = self.build()
        empty = self.by_condition(records, "A")
        self.assertEqual(empty["auxiliary_message"], "")
        self.assertEqual(empty["auxiliary_sha256"], hashlib.sha256(b"").hexdigest())
        file_only = self.by_condition(records, "B")
        self.assertIn("pkg/mod.py", file_only["auxiliary_message"])
        self.assertIn("pkg/second.py", file_only["auxiliary_message"])
        self.assertNotIn("-----", file_only["auxiliary_message"])
        self.assertNotIn(MOD_ADDED, file_only["auxiliary_message"])

    def test_condition_c_windows_are_the_base_commit_files(self) -> None:
        records, _, source_root = self.build()
        message = self.by_condition(records, "C")["auxiliary_message"]
        self.assertIn("not produced by a tool call", message)
        parsed = self.sections(message)
        self.assertEqual([match.group("path") for match, _ in parsed], ["pkg/mod.py", "pkg/second.py"])
        for match, body in parsed:
            lines = (
                source_root / TASK_ID / match.group("path")
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(int(match.group("total")), len(lines))
            pairs = [
                (int(number), text)
                for number, text in re.findall(r"^(\d+): (.*)$", body, re.M)
            ]
            gaps = {(int(m.group("first")), int(m.group("last"))) for m in NOT_SHOWN.finditer(body)}
            self.assertTrue(pairs, f"{match.group('path')} rendered nothing")
            for number, text in pairs:
                self.assertEqual(text, lines[number - 1])
            shown = {number for number, _ in pairs}
            for first, last in gaps:
                self.assertFalse(shown & set(range(first, last + 1)))

    def test_a_far_apart_second_hunk_is_never_dropped(self) -> None:
        records, manifest, _ = self.build(pad=20, max_lines_per_file=50)
        window = self.window(records, "mod.py")
        covered = set()
        for interval in window["intervals"]:
            covered |= set(range(interval["first_line"], interval["last_line"] + 1))
        self.assertIn(3, covered, "the first hunk must be shown")
        self.assertIn(40, covered, "the second hunk must be shown even though it is far away")
        self.assertEqual(window["hunks"], [{"old_start": 1, "old_count": 3}, {"old_start": 40, "old_count": 3}])
        self.assertEqual(manifest["tasks"][0]["hunks_not_covered"], [])
        self.assertLess(window["padding"], 20)
        self.assertFalse(window["truncated"])
        self.assertEqual(window["padding"], 14)
        self.assertEqual(window["shown_lines"], 48)
        self.assertEqual(
            [(i["first_line"], i["last_line"]) for i in window["intervals"]], [(1, 17), (26, 56)]
        )

    def test_a_hunk_is_kept_even_when_it_exceeds_the_cap(self) -> None:
        records, manifest, _ = self.build(pad=20, max_lines_per_file=4)
        window = self.window(records, "mod.py")
        intervals = [(i["first_line"], i["last_line"]) for i in window["intervals"]]
        self.assertEqual(intervals, [(1, 3), (40, 42)])
        self.assertEqual(window["padding"], 0)
        self.assertTrue(window["truncated"], "over-cap files must say so")
        self.assertEqual(window["shown_lines"], 6)
        self.assertEqual(manifest["tasks"][0]["hunks_not_covered"], [])

    def test_no_added_line_is_shown_and_undisclosed_overlap_is_reported(self) -> None:
        records, manifest, _ = self.build(include_marker=False)
        message = self.by_condition(records, "C")["auxiliary_message"]
        self.assertNotIn(MOD_ADDED, message)
        self.assertNotIn(SECOND_ADDED, message)
        self.assertEqual(manifest["tasks"][0]["added_line_disclosure"]["C"], [])
        self.assertEqual(manifest["tasks"][0]["added_line_disclosure"]["B"], [])

    def test_pre_existing_added_line_text_is_disclosed_with_its_base_line(self) -> None:
        records, manifest, _ = self.build(include_marker=True)
        message = self.by_condition(records, "C")["auxiliary_message"]
        self.assertIn(SECOND_ADDED, message)
        disclosure = manifest["tasks"][0]["added_line_disclosure"]["C"]
        entry = next(item for item in disclosure if item["text"] == SECOND_ADDED)
        self.assertEqual(entry["in_window_at"], [{"path": "pkg/second.py", "base_line": 5}])
        self.assertEqual(self.by_condition(records, "C")["added_line_disclosure"], disclosure)

    def test_total_line_budget_skips_the_rest_and_records_it(self) -> None:
        _, manifest, _ = self.build(max_total_lines=7, max_lines_per_file=30, pad=0)
        entry = manifest["tasks"][0]
        self.assertEqual(entry["files_with_windows"], ["pkg/mod.py"])
        self.assertEqual(
            entry["skipped"], [{"path": "pkg/second.py", "reason": "total line budget reached"}]
        )
        # The skipped file's hunk is reported rather than silently absent.
        self.assertEqual(
            entry["hunks_not_covered"], [{"path": "pkg/second.py", "old_start": 1, "old_end": 2}]
        )

    def test_missing_source_is_reported_rather_than_invented(self) -> None:
        records, manifest, _ = self.build(seed_source=False)
        self.assertEqual(manifest["tasks"][0]["files_with_windows"], [])
        reasons = {item["reason"] for item in manifest["tasks"][0]["skipped"]}
        self.assertEqual(reasons, {"source unavailable at the base commit"})
        self.assertEqual(manifest["tasks"][0]["hunks_not_covered_count"], 3)
        message = self.by_condition(records, "C")["auxiliary_message"]
        self.assertNotIn("-----", message)
        # C stays a superset of B: the unreadable files are still named.
        for path in ("pkg/mod.py", "pkg/second.py"):
            self.assertIn(path, message)
            self.assertIn(path, self.by_condition(records, "B")["auxiliary_message"])

    def test_manifest_hashes_agree_with_the_records(self) -> None:
        records, manifest, _ = self.build()
        entry = manifest["tasks"][0]
        for record in records:
            digest = hashlib.sha256(record["auxiliary_message"].encode("utf-8")).hexdigest()
            self.assertEqual(record["auxiliary_sha256"], digest)
            self.assertEqual(entry["auxiliary_sha256"][record["condition"]], digest)
        self.assertTrue(manifest["run_complete"])
        self.assertEqual(
            manifest["output_sha256"],
            hashlib.sha256((self.root / "oracle.jsonl").read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
