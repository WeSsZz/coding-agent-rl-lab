from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from coding_agent_rl_lab.dataset import DatasetError, load_tasks


class DatasetTests(unittest.TestCase):
    def test_builtin_dataset_is_versioned_and_unique(self) -> None:
        root = Path(__file__).resolve().parents[1]
        tasks = load_tasks(root / "datasets" / "development" / "tasks.jsonl")
        self.assertEqual(len(tasks), 2)
        self.assertEqual(len({task.task_id for task in tasks}), 2)
        self.assertTrue(all(task.base_commit == "fixture-v1" for task in tasks))

    def test_duplicate_task_ids_are_rejected(self) -> None:
        row = (
            '{"task_id":"duplicate","issue":"x","fixture_path":"x",'
            '"base_commit":"v1","test_command":["{python}","-V"],'
            '"split":"development","provenance":"test"}'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"
            path.write_text(f"{row}\n{row}\n", encoding="utf-8")
            with self.assertRaises(DatasetError):
                load_tasks(path)


if __name__ == "__main__":
    unittest.main()
