"""Pin the hunk parsing behind the tool-driven repair self-check.

The self-check is only meaningful if the replacement text it feeds to `replace_lines` is the hunk's
*new side* - context lines and added lines together. Passing only the added lines would delete the
surrounding code, the edit would be refused or the tests would fail, and the check would report the
protocol as broken when the harness was.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_repair_through_tools.py"
_spec = importlib.util.spec_from_file_location("check_repair_through_tools", SCRIPT)
assert _spec is not None and _spec.loader is not None
check = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = check
_spec.loader.exec_module(check)

PATCH = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -10,4 +10,5 @@ def alpha():
 context a
-removed b
+added b
 context c
 context d
@@ -40,2 +41,2 @@ def beta():
 old tail
-old line
+new line
diff --git a/pkg/second.py b/pkg/second.py
--- a/pkg/second.py
+++ b/pkg/second.py
@@ -1,2 +1,2 @@
 keep
-replace me
+replacement
"""


class HunkParsingTests(unittest.TestCase):
    def test_new_side_keeps_the_context_lines(self) -> None:
        hunks = check.patch_hunks_with_bodies(PATCH)
        self.assertEqual(sorted(hunks), ["pkg/mod.py", "pkg/second.py"])
        first = hunks["pkg/mod.py"][0]
        self.assertEqual((first["old_start"], first["old_count"]), (10, 4))
        self.assertEqual(first["new_lines"], ["context a", "added b", "context c", "context d"])
        self.assertNotIn("removed b", first["new_lines"])
        self.assertNotIn("old line", hunks["pkg/mod.py"][1]["new_lines"])
        self.assertEqual(hunks["pkg/mod.py"][1]["new_lines"], ["old tail", "new line"])
        self.assertEqual(hunks["pkg/second.py"][0]["new_lines"], ["keep", "replacement"])

    def test_the_old_range_matches_the_number_of_replaced_lines(self) -> None:
        hunks = check.patch_hunks_with_bodies(PATCH)
        for path, entries in hunks.items():
            for entry in entries:
                replaced = entry["old_start"] + entry["old_count"] - 1
                self.assertGreaterEqual(replaced, entry["old_start"], path)

    def test_read_window_covers_the_hunk_and_meets_the_tool_minimum(self) -> None:
        start, end = check.read_window({"old_start": 10, "old_count": 4})
        self.assertLessEqual(start, 10)
        self.assertGreaterEqual(end, 13)
        self.assertGreaterEqual(end - start + 1, 20)
        start, end = check.read_window({"old_start": 1, "old_count": 1})
        self.assertEqual(start, 1)
        self.assertGreaterEqual(end - start + 1, 20)

    def test_a_pure_insertion_is_visible_as_such(self) -> None:
        patch = """--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -5,0 +6,2 @@
+brand new
+second new
"""
        hunks = check.patch_hunks_with_bodies(patch)
        self.assertEqual(hunks["pkg/mod.py"][0]["old_count"], 0)
        self.assertEqual(hunks["pkg/mod.py"][0]["new_lines"], ["brand new", "second new"])


if __name__ == "__main__":
    unittest.main()
