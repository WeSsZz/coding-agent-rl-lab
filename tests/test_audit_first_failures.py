"""Pin the first-failure audit's two load-bearing classifications.

The audit exists to tell a *protocol* failure (the policy rewrote a range it never read, or repeated a
read until the guard stopped it) from a *capability* failure (the policy read the code and wrote
something that does not compile). If those two classifications are wrong the audit's conclusion is
wrong, so they are tested directly.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_first_failures.py"
_spec = importlib.util.spec_from_file_location("audit_first_failures", SCRIPT)
assert _spec is not None and _spec.loader is not None
audit_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = audit_module
_spec.loader.exec_module(audit_module)

GOLD = {"task": "+++ b/pkg/mod.py\n" + "+++ b/pkg/other.py\n"}
SOURCE = "\n".join(f"line {number}" for number in range(1, 121)) + "\n"


def step(kind, arguments, observation, sequence=1):
    return {
        "sequence": sequence,
        "action": {"kind": kind, "arguments": arguments},
        "observation": observation,
        "test_result": None,
        "policy_metadata": {},
    }


def trajectory(steps, *, changed=(), failed_nodes=(), regressed=0):
    return {
        "task_id": "task",
        "repetition": 1,
        "seed": 1,
        "policy": {"metadata": {"diagnostic_condition": "A"}},
        "steps": steps,
        "changed_files": list(changed),
        "verifier": {
            "failed_nodes": list(failed_nodes),
            "pass_to_pass_regressed": regressed,
            "fail_to_pass_total": 1,
            "fail_to_pass_resolved": 0,
        },
        "reward": {"steps": len(steps)},
    }


class ClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = TemporaryDirectory()
        self.source_root = Path(self._temporary.name)
        (self.source_root / "task" / "pkg").mkdir(parents=True)
        (self.source_root / "task" / "pkg" / "mod.py").write_text(SOURCE, encoding="utf-8")

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def classify(self, steps, index):
        return audit_module.classify_unparseable(
            trajectory(steps), steps[index], self.source_root, {"pkg/mod.py"}
        )

    def test_indentation_is_recorded_as_the_proximate_cause(self) -> None:
        steps = [
            step("read_file", {"path": "pkg/mod.py", "start_line": 1, "end_line": 120}, "1: line 1"),
            step(
                "replace_text",
                {"path": "pkg/mod.py", "old": "line 40", "new": "if x:\n    pass"},
                "Tool error: it would leave pkg/mod.py unparseable (IndentationError)",
                sequence=2,
            ),
        ]
        result = self.classify(steps, 1)
        self.assertTrue(result["first_line_unindented"])
        self.assertEqual(result["target_range"], "read")
        self.assertTrue(result["target_in_a_repair_file"])

    def test_a_range_the_policy_never_saw_is_recorded_next_to_the_indentation(self) -> None:
        steps = [
            step("read_file", {"path": "pkg/mod.py", "start_line": 1, "end_line": 20}, "1: line 1"),
            step(
                "replace_lines",
                {"path": "pkg/mod.py", "start_line": 110, "end_line": 112, "new": "        x = 1"},
                "Tool error: it would leave pkg/mod.py unparseable (IndentationError)",
                sequence=2,
            ),
        ]
        result = self.classify(steps, 1)
        self.assertFalse(result["first_line_unindented"])
        self.assertEqual(result["target_range"], "never_read")
        self.assertEqual(result["target"], (110, 112))

    def test_an_edit_a_repair_file_outside_the_read_set_is_still_located(self) -> None:
        steps = [
            step("read_file", {"path": "pkg/other.py", "start_line": 1, "end_line": 5}, "1: x"),
            step(
                "replace_lines",
                {"path": "pkg/mod.py", "start_line": 110, "end_line": 112, "new": "        x = 1"},
                "Tool error: it would leave pkg/mod.py unparseable (IndentationError)",
                sequence=2,
            ),
        ]
        result = self.classify(steps, 1)
        self.assertEqual(result["target_range"], "file_never_read")
        self.assertEqual(result["target"], (110, 112))


class AuditShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = TemporaryDirectory()
        self.source_root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_records_the_first_refusal_and_the_applied_edit_together(self) -> None:
        steps = [
            step("read_file", {"path": "pkg/mod.py"}, "1: line 1"),
            step("read_file", {"path": "pkg/mod.py"}, "Tool error: do not reread an unchanged file", sequence=2),
            step(
                "replace_text",
                {"path": "pkg/mod.py", "old": "line 1", "new": "line 1 changed"},
                "Updated pkg/mod.py.",
                sequence=3,
            ),
            step(
                "replace_text",
                {"path": "pkg/elsewhere.py", "old": "a", "new": "b"},
                "Updated pkg/elsewhere.py.",
                sequence=4,
            ),
        ]
        record = audit_module.audit(
            trajectory(steps, changed=("pkg/mod.py",), failed_nodes=("tests/x.py::t",), regressed=2),
            gold_paths={"pkg/mod.py"},
            source_root=self.source_root,
        )
        self.assertEqual(record["first_refusal"]["step"], 2)
        self.assertIn(audit_module.REREAD, record["first_refusal"]["refusal"])
        self.assertEqual(record["first_refusal"]["answered"]["kind"], "read_file")
        self.assertEqual(record["edits_in_a_repair_file"], ["pkg/mod.py"])
        self.assertEqual(record["edits_outside_the_repair"], ["pkg/elsewhere.py"])
        self.assertEqual(record["failed_nodes"], ["t"])
        self.assertEqual(record["pass_to_pass_regressed"], 2)

    def test_gold_paths_come_from_the_patch_headers(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "gold.json"
            path.write_text(json.dumps(GOLD), encoding="utf-8")
            self.assertEqual(
                audit_module.gold_paths_by_task(path), {"task": {"pkg/mod.py", "pkg/other.py"}}
            )


if __name__ == "__main__":
    unittest.main()
