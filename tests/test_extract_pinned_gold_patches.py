"""Pin the gold-patch extractor: it reads the shard, and it refuses to invent a missing patch.

Skipped where pyarrow is absent, which is the rollout VM by design - the extractor is only meant to
run on the machine that can read the published shard.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_pinned_gold_patches.py"

try:
    import pyarrow  # noqa: F401

    HAVE_PYARROW = True
except ModuleNotFoundError:
    HAVE_PYARROW = False

PATCH = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,2 +1,3 @@
 keep
+added
"""
OTHER_PATCH = """diff --git a/pkg/other.py b/pkg/other.py
--- a/pkg/other.py
+++ b/pkg/other.py
@@ -1,1 +1,2 @@
+added too
"""


@unittest.skipUnless(HAVE_PYARROW, "pyarrow is required to write the test shard")
class ExtractPinnedGoldPatchesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.parquet = self.root / "rows.parquet"

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def write_shard(self, rows: list[dict]) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, self.parquet)

    def run_script(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            (sys.executable, str(SCRIPT), *arguments), capture_output=True, text=True
        )

    def test_keeps_the_pinned_order_and_writes_only_the_requested_rows(self) -> None:
        self.write_shard(
            [
                {"instance_id": "getmoto__moto-7514", "patch": OTHER_PATCH},
                {"instance_id": "getmoto__moto-7365", "patch": PATCH},
                {"instance_id": "getmoto__moto-9999", "patch": "unrelated"},
            ]
        )
        output = self.root / "gold.json"
        completed = self.run_script(
            "--parquet",
            str(self.parquet),
            "--task-id",
            "getmoto__moto-7365",
            "--task-id",
            "getmoto__moto-7514",
            "--output",
            str(output),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(list(payload), ["getmoto__moto-7365", "getmoto__moto-7514"])
        self.assertEqual(payload["getmoto__moto-7365"], PATCH)
        self.assertNotIn("unrelated", output.read_text(encoding="utf-8"))

    def test_refuses_a_row_whose_patch_is_empty(self) -> None:
        self.write_shard([{"instance_id": "getmoto__moto-7365", "patch": "  "}])
        completed = self.run_script(
            "--parquet",
            str(self.parquet),
            "--task-id",
            "getmoto__moto-7365",
            "--output",
            str(self.root / "gold.json"),
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("no gold patch", completed.stderr)

    def test_refuses_an_instance_id_that_is_not_pinned(self) -> None:
        self.write_shard([{"instance_id": "getmoto__moto-7365", "patch": PATCH}])
        completed = self.run_script(
            "--parquet",
            str(self.parquet),
            "--task-id",
            "getmoto__moto-9999",
            "--output",
            str(self.root / "gold.json"),
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("not a pinned instance id", completed.stderr)


if __name__ == "__main__":
    unittest.main()
