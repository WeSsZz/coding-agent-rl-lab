from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_task_splits import (  # noqa: E402
    ROUTE_V1_TOUCHED,
    apply_cost_filter,
    assign,
    near_duplicate_groups,
    selection_key,
    text_hash,
    validate,
)


def _row(
    instance_id: str,
    *,
    repo: str = "acme/widget",
    base_commit: str = "0" * 40,
    patch: str | None = None,
    fail_to_pass: int = 1,
    pass_to_pass: int = 1,
) -> dict:
    # Distinct patches by default: an identical gold patch is a near-duplicate signal, so a
    # shared default would silently merge every synthetic row into one unit.
    return {
        "instance_id": instance_id,
        "repo": repo,
        "version": "1.0",
        "base_commit": base_commit,
        "patch": f"patch-{instance_id}" if patch is None else patch,
        "test_patch": "tp",
        "problem_statement": "ps",
        "FAIL_TO_PASS": [f"t{i}" for i in range(fail_to_pass)],
        "PASS_TO_PASS": [f"p{i}" for i in range(pass_to_pass)],
    }


class SelectionKeyTests(unittest.TestCase):
    def test_the_key_is_a_function_of_the_instance_and_the_salt(self) -> None:
        self.assertEqual(selection_key("a", "s"), selection_key("a", "s"))
        self.assertNotEqual(selection_key("a", "s"), selection_key("a", "t"))
        self.assertNotEqual(selection_key("a", "s"), selection_key("b", "s"))


class NearDuplicateTests(unittest.TestCase):
    def test_two_issues_on_one_commit_are_one_unit(self) -> None:
        rows = [_row("a", base_commit="c1"), _row("b", base_commit="c1"), _row("c", base_commit="c2")]

        groups = near_duplicate_groups(rows)

        self.assertEqual(groups["a"], groups["b"])
        self.assertNotIn("c", groups)

    def test_an_identical_gold_patch_is_one_unit_even_on_different_commits(self) -> None:
        rows = [
            _row("a", base_commit="c1", patch="same"),
            _row("b", base_commit="c2", patch="same"),
        ]

        groups = near_duplicate_groups(rows)

        self.assertEqual(groups["a"], groups["b"])

    def test_distinct_patches_on_distinct_commits_are_not_grouped(self) -> None:
        rows = [_row("a", base_commit="c1", patch="x"), _row("b", base_commit="c2", patch="y")]

        self.assertEqual(near_duplicate_groups(rows), {})

    def test_text_hash_is_stable(self) -> None:
        self.assertEqual(text_hash("abc"), text_hash("abc"))
        self.assertNotEqual(text_hash("abc"), text_hash("abd"))


class AssigmentTests(unittest.TestCase):
    def _pool(self, count: int = 12) -> list[dict]:
        return [_row(f"t{i:02d}", base_commit=f"c{i}") for i in range(count)]

    def test_assignment_is_deterministic(self) -> None:
        rows = self._pool()
        sizes = {"train": 4, "development": 0, "final-test": 4}

        first, _ = assign(rows, sizes, salt="s")
        second, _ = assign(rows, sizes, salt="s")

        self.assertEqual(first, second)

    def test_a_different_salt_reorders_the_split(self) -> None:
        rows = self._pool(40)
        sizes = {"train": 6, "development": 0, "final-test": 6}

        first, _ = assign(rows, sizes, salt="s1")
        second, _ = assign(rows, sizes, salt="s2")

        self.assertNotEqual(first["train"], second["train"])

    def test_splits_do_not_overlap(self) -> None:
        rows = self._pool(20)
        sizes = {"train": 6, "development": 0, "final-test": 6}

        allocations, _ = assign(rows, sizes, salt="s")

        self.assertFalse(set(allocations["train"]) & set(allocations["final-test"]))

    def test_a_near_duplicate_group_is_never_split(self) -> None:
        rows = [_row(f"u{i:02d}", base_commit=f"c{i}") for i in range(10)]
        rows += [_row("d1", base_commit="shared"), _row("d2", base_commit="shared")]
        sizes = {"train": 5, "development": 0, "final-test": 5}

        allocations, _ = assign(rows, sizes, salt="s")

        where = {item: name for name, members in allocations.items() for item in members}
        self.assertEqual(where.get("d1"), where.get("d2"))

    def test_a_route_v1_instance_is_always_development(self) -> None:
        migrated = next(iter(ROUTE_V1_TOUCHED))
        rows = [_row(migrated, base_commit="legacy")] + [_row(f"n{i}", base_commit=f"c{i}") for i in range(8)]
        sizes = {"train": 3, "development": 0, "final-test": 3}

        allocations, stats = assign(rows, sizes, salt="s")

        self.assertIn(migrated, allocations["development"])
        self.assertNotIn(migrated, allocations["final-test"])
        self.assertEqual(stats["development_from_route_v1"], 1)

    def test_sizes_are_respected_when_the_pool_allows(self) -> None:
        rows = self._pool(30)
        sizes = {"train": 8, "development": 0, "final-test": 5}

        allocations, _ = assign(rows, sizes, salt="s")

        self.assertEqual(len(allocations["train"]), 8)
        self.assertEqual(len(allocations["final-test"]), 5)


class CostFilterTests(unittest.TestCase):
    def test_expensive_instances_are_dropped_and_counted(self) -> None:
        rows = [
            _row("cheap", pass_to_pass=10, fail_to_pass=2),
            _row("wide", pass_to_pass=500),
            _row("deep", fail_to_pass=50),
        ]

        kept, dropped = apply_cost_filter(rows, max_pass_to_pass=80, max_fail_to_pass=8)

        self.assertEqual([row["instance_id"] for row in kept], ["cheap"])
        self.assertEqual(dropped, {"pass_to_pass_over_cap": 1, "fail_to_pass_over_cap": 1})

    def test_an_empty_result_is_an_error_rather_than_an_empty_split(self) -> None:
        from scripts.build_task_splits import SplitError

        with self.assertRaises(SplitError):
            apply_cost_filter([_row("wide", pass_to_pass=900)], max_pass_to_pass=80, max_fail_to_pass=8)


class ValidationTests(unittest.TestCase):
    def test_a_single_repository_final_test_is_refused(self) -> None:
        rows = [
            _row("a", repo="one/repo", base_commit="c1"),
            _row("b", repo="one/repo", base_commit="c2"),
        ]
        allocations = {"train": [], "development": [], "final-test": ["a", "b"]}

        problems = validate(allocations, rows, sizes={"train": 0, "development": 0, "final-test": 2})

        self.assertTrue(any("single-repository" in problem for problem in problems), problems)

    def test_a_route_v1_task_in_the_final_test_is_refused(self) -> None:
        migrated = next(iter(ROUTE_V1_TOUCHED))
        rows = [_row(migrated, base_commit="c1"), _row("b", repo="two/repo", base_commit="c2")]
        allocations = {"train": [], "development": [], "final-test": [migrated, "b"]}

        problems = validate(allocations, rows, sizes={"train": 0, "development": 0, "final-test": 2})

        self.assertTrue(any("route-v1" in problem for problem in problems), problems)

    def test_an_underfilled_split_is_refused(self) -> None:
        rows = [_row("a", repo="one/repo"), _row("b", repo="two/repo")]
        allocations = {
            "train": ["a"],
            "development": [],
            "final-test": ["b"],
        }

        problems = validate(allocations, rows, sizes={"train": 9, "development": 0, "final-test": 1})

        self.assertTrue(any("train: got 1" in problem for problem in problems), problems)

    def test_a_clean_split_passes(self) -> None:
        rows = [
            _row("a", repo="one/repo", base_commit="c1"),
            _row("b", repo="two/repo", base_commit="c2"),
            _row("c", repo="three/repo", base_commit="c3"),
            _row("d", repo="four/repo", base_commit="c4"),
        ]
        allocations = {"train": ["a"], "development": [], "final-test": ["b", "c"]}

        problems = validate(allocations, rows, sizes={"train": 1, "development": 0, "final-test": 2})

        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
