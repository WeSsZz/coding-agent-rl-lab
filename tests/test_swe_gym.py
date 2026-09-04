from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_agent_rl_lab.contracts import DatasetSplit
from coding_agent_rl_lab.swe_gym import (
    SWE_GYM_ENVIRONMENT_REVISION,
    SWEGymAdapterError,
    SWEGymTaskAdapter,
    audited_swe_gym_test_command,
    load_swe_gym_jsonl,
)
from coding_agent_rl_lab.swe_gym_smoke import (
    PINNED_DEVELOPMENT_ROWS,
    SWE_GYM_TASK_SET_ROWS,
    dataset_split_for_task_set,
    download_pinned_rows,
    load_or_download_pinned_rows,
    select_pinned_rows,
)


def _sample_row() -> dict[str, object]:
    return {
        "instance_id": "getmoto__moto-7365",
        "problem_statement": "DynamoDB update_item should use decimal arithmetic.",
        "repo": "getmoto/moto",
        "base_commit": "7f6c9cb1deafb280fe7fcc7551c38e397f11a706",
        "version": "5.0",
        "created_at": "2024-02-19 20:29:03",
        "patch": "gold solution must not reach the policy",
        "test_patch": (
            "diff --git a/tests/test_bug.py b/tests/test_bug.py\n"
            "--- a/tests/test_bug.py\n"
            "+++ b/tests/test_bug.py\n"
        ),
        "FAIL_TO_PASS": ["tests/test_bug.py::test_decimal"],
        "PASS_TO_PASS": ["tests/test_existing.py::test_regression"],
    }


class SWEGymAdapterTests(unittest.TestCase):
    def test_pinned_development_set_has_ten_unique_answer_free_identifiers(self) -> None:
        self.assertEqual(len(PINNED_DEVELOPMENT_ROWS), 10)
        self.assertEqual(len({item.offset for item in PINNED_DEVELOPMENT_ROWS}), 10)
        self.assertEqual(len({item.instance_id for item in PINNED_DEVELOPMENT_ROWS}), 10)
        self.assertTrue(all(len(item.base_commit) == 40 for item in PINNED_DEVELOPMENT_ROWS))

    def test_train_regression_and_held_out_sets_are_disjoint_and_exhaustive(self) -> None:
        named_sets = tuple(
            {item.instance_id for item in SWE_GYM_TASK_SET_ROWS[name]}
            for name in ("train", "regression", "held-out")
        )

        self.assertEqual(tuple(map(len, named_sets)), (6, 2, 2))
        self.assertEqual(SWE_GYM_TASK_SET_ROWS["train"][0].instance_id, "getmoto__moto-7509")
        self.assertFalse(named_sets[0] & named_sets[1])
        self.assertFalse(named_sets[0] & named_sets[2])
        self.assertFalse(named_sets[1] & named_sets[2])
        self.assertEqual(set.union(*named_sets), {item.instance_id for item in PINNED_DEVELOPMENT_ROWS})
        self.assertEqual(dataset_split_for_task_set("train"), DatasetSplit.DEVELOPMENT)
        self.assertEqual(dataset_split_for_task_set("regression"), DatasetSplit.REGRESSION)
        self.assertEqual(dataset_split_for_task_set("held-out"), DatasetSplit.HELD_OUT)

    def test_cached_held_out_rows_are_selected_by_identity_not_prefix(self) -> None:
        cached_rows = []
        for pinned in reversed(PINNED_DEVELOPMENT_ROWS):
            row = _sample_row()
            row["instance_id"] = pinned.instance_id
            row["base_commit"] = pinned.base_commit
            row.pop("patch")
            cached_rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in cached_rows),
                encoding="utf-8",
            )

            rows = load_or_download_pinned_rows(path, limit=2, task_set="held-out")

        self.assertEqual(
            tuple(row["instance_id"] for row in rows),
            tuple(item.instance_id for item in SWE_GYM_TASK_SET_ROWS["held-out"]),
        )

    def test_exact_task_ids_preserve_requested_order_and_enforce_split(self) -> None:
        selected = select_pinned_rows(
            "train",
            task_ids=("getmoto__moto-7365", "getmoto__moto-7509"),
        )

        self.assertEqual(
            tuple(item.instance_id for item in selected),
            ("getmoto__moto-7365", "getmoto__moto-7509"),
        )
        with self.assertRaisesRegex(ValueError, "outside task set train"):
            select_pinned_rows("train", task_ids=("getmoto__moto-7393",))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            select_pinned_rows(
                "train",
                task_ids=("getmoto__moto-7509", "getmoto__moto-7509"),
            )
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            select_pinned_rows("train", limit=1, task_ids=("getmoto__moto-7509",))

    @patch("coding_agent_rl_lab.swe_gym_smoke._download_pinned_row")
    def test_rollout_download_removes_gold_patch_and_hints(self, download) -> None:
        row = _sample_row()
        row["hints_text"] = "answer-like hint"
        download.return_value = row

        sanitized = download_pinned_rows(limit=1)[0]

        self.assertNotIn("patch", sanitized)
        self.assertNotIn("hints_text", sanitized)
        self.assertIn("test_patch", sanitized)

    def test_audited_moto_command_uses_pinned_environment_and_declared_tests(self) -> None:
        command = audited_swe_gym_test_command(_sample_row())
        self.assertEqual(command[:4], (
            "/opt/miniconda3/envs/testbed/bin/pytest",
            "-n0",
            "-rA",
            "--",
        ))
        self.assertEqual(command[4:], (
            "tests/test_bug.py::test_decimal",
            "tests/test_existing.py::test_regression",
        ))
        self.assertEqual(len(SWE_GYM_ENVIRONMENT_REVISION), 40)

    def test_unknown_repo_version_has_no_guessed_test_command(self) -> None:
        row = _sample_row()
        row["version"] = "unknown"
        with self.assertRaisesRegex(SWEGymAdapterError, "no audited test command"):
            audited_swe_gym_test_command(row)

    def test_official_row_maps_to_public_task_and_private_environment_spec(self) -> None:
        bundle = SWEGymTaskAdapter().adapt(
            _sample_row(),
            split=DatasetSplit.DEVELOPMENT,
            test_command=("python", "-m", "pytest", "-q", "tests/test_bug.py::test_decimal"),
        )

        self.assertIsNone(bundle.task.fixture_path)
        self.assertEqual(bundle.task.base_commit, "7f6c9cb1deafb280fe7fcc7551c38e397f11a706")
        self.assertEqual(
            bundle.environment.image,
            "xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365:latest",
        )
        self.assertEqual(bundle.environment.repository_path, "/testbed")
        self.assertNotIn("patch", bundle.task.metadata)
        self.assertNotIn("test_patch", bundle.task.metadata)
        self.assertNotIn("FAIL_TO_PASS", bundle.task.metadata)
        self.assertIn("test_bug.py", bundle.environment.test_patch)

    def test_raw_row_requires_an_explicit_versioned_test_command(self) -> None:
        with self.assertRaisesRegex(SWEGymAdapterError, "portable test command"):
            SWEGymTaskAdapter().adapt(_sample_row(), split=DatasetSplit.DEVELOPMENT)

    def test_invalid_schema_is_rejected(self) -> None:
        row = _sample_row()
        del row["base_commit"]
        with self.assertRaisesRegex(SWEGymAdapterError, "base_commit"):
            SWEGymTaskAdapter().adapt(
                row,
                split=DatasetSplit.DEVELOPMENT,
                test_command=("python", "-m", "pytest"),
            )

    def test_jsonl_loader_accepts_audited_enriched_rows(self) -> None:
        row = _sample_row()
        row["test_command"] = ["python", "-m", "pytest", "-q", "tests/test_bug.py::test_decimal"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "swe-gym.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            bundles = load_swe_gym_jsonl(path, split=DatasetSplit.HELD_OUT)
        self.assertEqual(len(bundles), 1)
        self.assertEqual(bundles[0].task.split, DatasetSplit.HELD_OUT)


if __name__ == "__main__":
    unittest.main()
