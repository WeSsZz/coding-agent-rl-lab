"""Freeze a train / development / final-test split over the official SWE-Gym task pool.

Why this is a script and not a hand-written list:

* Selection must be reproducible from the published parquet alone. The order is
  `sha256(salt + instance_id)`, not a shuffle with a seed nobody can re-derive.
* Near-duplicates are the real leakage risk, not repeated `task_id`s. Two instances that
  share `(repo, base_commit)` are different issues on one commit, and two instances that
  share an identical gold `patch` are the same fix. Both are forced into the same split.
* Every instance the route-v1 work already touched is pinned to `development`, so no
  diagnostic task can reappear in the final test.
* A repository/version pair whose test invocation has not been audited is recorded as
  `test_command_status: unaudited`; the split is still written so the budget can be
  planned, but such an instance is not runnable until `swe_gym.py`'s audited table covers it.

The counts are provisional: they are capped by image pulls and the measured rollout
throughput, both of which are unknown until the environment work in phase 2 is done. The
manifest says so instead of pretending the numbers are already justified.

Usage:
    python scripts/build_task_splits.py --shard work/private/swe-gym-train-rows.parquet \
        --output manifests/route-v2-task-split.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

#: Default pool: one pinned version per repository, chosen so a single audited pytest
#: invocation covers every instance taken from that repository.
DEFAULT_SOURCES: tuple[tuple[str, str], ...] = (
    ("getmoto/moto", "5.0"),
    ("conan-io/conan", "2.0"),
    ("pydantic/pydantic", "2.7"),
)

#: Everything route v1 touched. The A/B/C diagnostic, the gold-derived SFT set and the
#: single-step GRPO runs all drew from these, so they are development tasks forever.
ROUTE_V1_TOUCHED: dict[str, str] = {
    "getmoto__moto-7509": "route v1 train task, single-step GRPO",
    "getmoto__moto-7365": "route v1 train task and A/B/C diagnostic condition task",
    "getmoto__moto-7514": "route v1 train task and A/B/C diagnostic condition task",
    "getmoto__moto-7646": "route v1 train task and navigation/SFT search",
    "getmoto__moto-7446": "route v1 train task",
    "getmoto__moto-7607": "route v1 train task, seven frozen pre-edit states",
    "getmoto__moto-7385": "route v1 regression task",
    "getmoto__moto-7608": "route v1 regression task and A/B/C diagnostic condition task",
    "getmoto__moto-7393": "repeatedly inspected in route v1; never a held-out claim again",
    "getmoto__moto-7537": "route v1 held-out task; kept as development to be safe",
}

DEFAULT_SALT = "coding-agent-rl-lab/route-v2/task-split/2026-09-22"


class SplitError(RuntimeError):
    pass


def selection_key(instance_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}/{instance_id}".encode()).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_shard(path: Path, sources: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on the host
        raise SplitError("pyarrow is required to read the official shard") from exc

    wanted = set(sources)
    rows = [row for row in pq.read_table(path).to_pylist() if (row["repo"], row["version"]) in wanted]
    if not rows:
        raise SplitError(f"no rows matched the requested repo/version pairs in {path}")
    return rows


def apply_cost_filter(
    rows: list[dict[str, Any]], *, max_pass_to_pass: int, max_fail_to_pass: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop instances whose verifier run is too expensive or too diffuse to grade cheaply.

    This is a budget rule declared before any model is trained, not a post-hoc filter: the
    verifier runs every PASS_TO_PASS node on every evaluation, so a task with over a thousand
    of them costs more wall-clock than a hundred tasks with a handful.
    """

    kept: list[dict[str, Any]] = []
    dropped = {"pass_to_pass_over_cap": 0, "fail_to_pass_over_cap": 0}
    for row in rows:
        if len(row["PASS_TO_PASS"]) > max_pass_to_pass:
            dropped["pass_to_pass_over_cap"] += 1
            continue
        if len(row["FAIL_TO_PASS"]) > max_fail_to_pass:
            dropped["fail_to_pass_over_cap"] += 1
            continue
        kept.append(row)
    if not kept:
        raise SplitError("the cost filter removed every instance")
    return kept, dropped


def apply_image_availability_filter(
    rows: list[dict[str, Any]], unavailable_images: set[str]
) -> tuple[list[dict[str, Any]], int]:
    """Drop instances whose published image does not exist.

    The official SWE-Gym data includes rows whose instance image was never published. Leaving
    one in a split is not a cosmetic problem: the container start fails, and that lands in the
    results as an infrastructure failure, which this project is careful never to score as a
    policy failure. Availability is produced by `scripts/vm/check_images_exist.sh`, which asks
    the registry directly rather than pulling.
    """

    if not unavailable_images:
        return rows, 0
    kept = [row for row in rows if image_name(row) not in unavailable_images]
    removed = len(rows) - len(kept)
    if not kept:
        raise SplitError("the image availability filter removed every instance")
    return kept, removed


def image_name(row: dict[str, Any]) -> str:
    return (
        "xingyaoww/sweb.eval.x86_64."
        + row["instance_id"].replace("__", "_s_").lower()
        + ":latest"
    )


def near_duplicate_groups(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Map an instance id to a group id that must land in a single split."""

    groups: dict[str, list[str]] = {}
    by_commit: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_patch: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_commit[(row["repo"], row["base_commit"])].append(row["instance_id"])
        by_patch[text_hash(row["patch"])].append(row["instance_id"])

    for members in list(by_commit.values()) + list(by_patch.values()):
        if len(members) < 2:
            continue
        group_id = "|".join(sorted(members))
        for member in members:
            groups[member] = group_id
    return groups


def assign(
    rows: list[dict[str, Any]],
    sizes: dict[str, int],
    *,
    salt: str,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    groups = near_duplicate_groups(rows)
    by_id = {row["instance_id"]: row for row in rows}

    # A group is scheduled as one unit at the position of its lowest-keyed member, so the
    # split is a deterministic function of the pool and the salt.
    units: dict[str, list[str]] = {}
    for instance_id in sorted(by_id, key=lambda item: selection_key(item, salt)):
        unit_id = groups.get(instance_id, instance_id)
        units.setdefault(unit_id, [])
        if instance_id not in units[unit_id]:
            units[unit_id].append(instance_id)

    ordered_units = sorted(
        units.values(),
        key=lambda members: min(selection_key(member, salt) for member in members),
    )

    allocations: dict[str, list[str]] = {name: [] for name in sizes}
    # Development is the route-v1 set plus anything that duplicates it; everything else is
    # dealt out to train first and final-test last, so an unexpectedly large pool never
    # starves the test set.
    development = sorted(
        instance_id
        for instance_id in by_id
        if instance_id in ROUTE_V1_TOUCHED or groups.get(instance_id) in ROUTE_V1_TOUCHED
    )
    allocations["development"] = development
    development_set = set(development)

    order = ("train", "final-test")
    remaining = [unit for unit in ordered_units if not set(unit) & development_set]
    cursor = 0
    for name in order:
        target = sizes[name]
        while cursor < len(remaining) and len(allocations[name]) < target:
            allocations[name].extend(remaining[cursor])
            cursor += 1
        allocations[name] = sorted(allocations[name])

    stats = {
        "pool_instances": len(rows),
        "pool_units": len(ordered_units),
        "pool_near_duplicate_groups": len(set(groups.values())),
        "pool_instances_in_a_duplicate_group": len(groups),
        "unplaced_units": len(remaining) - cursor,
        "development_from_route_v1": sum(1 for item in development if item in ROUTE_V1_TOUCHED),
    }
    return allocations, stats


def validate(
    allocations: dict[str, list[str]],
    rows: list[dict[str, Any]],
    *,
    sizes: dict[str, int],
) -> list[str]:
    problems: list[str] = []
    by_id = {row["instance_id"]: row for row in rows}
    groups = near_duplicate_groups(rows)

    for name, members in allocations.items():
        missing = [item for item in members if item not in by_id]
        if missing:
            problems.append(f"{name}: {len(missing)} id(s) are not in the pool, e.g. {missing[:3]}")
        if len(set(members)) != len(members):
            problems.append(f"{name}: contains a repeated instance id")

    seen: dict[str, str] = {}
    for name, members in allocations.items():
        for member in members:
            if member in seen:
                problems.append(f"{member} is in both {seen[member]} and {name}")
            seen[member] = name

    for member, unit in groups.items():
        if member not in seen:
            continue
        others = {seen.get(other) for other in unit.split("|") if other in seen}
        if len(others) > 1:
            problems.append(f"near-duplicate group {unit} spans splits {sorted(others)}")

    if len(allocations.get("final-test", [])) > 0:
        repos = {by_id[item]["repo"] for item in allocations["final-test"] if item in by_id}
        if len(repos) < 2:
            problems.append(
                f"final-test covers only {sorted(repos)}; a single-repository test set cannot "
                "support a generalisation claim"
            )

    for name in ("train", "development", "final-test"):
        if len(allocations.get(name, [])) < sizes.get(name, 0):
            problems.append(
                f"{name}: got {len(allocations.get(name, []))} of the requested {sizes[name]}"
            )

    for member in allocations.get("final-test", []):
        if member in ROUTE_V1_TOUCHED:
            problems.append(f"final-test contains route-v1 task {member}")
    return problems


def build_manifest(
    rows: list[dict[str, Any]],
    allocations: dict[str, list[str]],
    stats: dict[str, Any],
    *,
    sizes: dict[str, int],
    salt: str,
    shard: Path,
    sources: tuple[tuple[str, str], ...],
    audited: set[tuple[str, str]],
    max_pass_to_pass: int,
    max_fail_to_pass: int,
) -> dict[str, Any]:
    by_id = {row["instance_id"]: row for row in rows}
    shard_sha = hashlib.sha256(shard.read_bytes()).hexdigest() if shard.is_file() else None

    instances = []
    for name in ("train", "development", "final-test"):
        for instance_id in allocations[name]:
            row = by_id[instance_id]
            instances.append(
                {
                    "instance_id": instance_id,
                    "split": name,
                    "repo": row["repo"],
                    "version": row["version"],
                    "base_commit": row["base_commit"],
                    "fail_to_pass_count": len(row["FAIL_TO_PASS"]),
                    "pass_to_pass_count": len(row["PASS_TO_PASS"]),
                    "gold_patch_sha256": text_hash(row["patch"]),
                    "test_patch_sha256": text_hash(row["test_patch"]),
                    "problem_statement_sha256": text_hash(row["problem_statement"]),
                    "test_command_status": (
                        "audited" if (row["repo"], row["version"]) in audited else "unaudited"
                    ),
                    "selection_key": selection_key(instance_id, salt)[:16],
                    "route_v1_note": ROUTE_V1_TOUCHED.get(instance_id),
                    "image": image_name(row),
                }
            )

    per_split_repos: dict[str, list[str]] = {
        name: sorted({by_id[item]["repo"] for item in allocations[name]})
        for name in ("train", "development", "final-test")
    }
    counts = Counter(item["split"] for item in instances)
    return {
        "schema_version": 1,
        "manifest_id": "route-v2-task-split",
        "generated_at": "2026-09-22",
        "status": "provisional",
        "provisional_until_throughput_measured": True,
        "why_provisional": (
            "The per-split counts are a budget guess. Image-pull cost and rollout throughput "
            "are measured in phase 2 and phase 5; the split rule (exclusions, near-duplicate "
            "grouping, determinism) is frozen now and must not change to chase a result."
        ),
        "source": {
            "dataset_id": "SWE-Gym/SWE-Gym",
            "dataset_revision": "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb",
            "local_shard": str(shard).replace("\\", "/"),
            "local_shard_sha256": shard_sha,
            "license": "mit",
            "repositories": [{"repo": repo, "version": version} for repo, version in sources],
        },
        "rule": {
            "ordering": "ascending sha256('<salt>/<instance_id>')",
            "salt": salt,
            "near_duplicate_rule": (
                "instances sharing (repo, base_commit) or an identical gold patch are one unit "
                "and are never split across train/development/final-test"
            ),
            "route_v1_rule": (
                "every instance route v1 touched is pinned to development; final-test may not "
                "contain one"
            ),
            "test_command_rule": (
                "a (repo, version) pair without an audited pytest invocation is marked unaudited "
                "and is not runnable until the audited table covers it"
            ),
            "cost_filter": (
                f"instances with more than {max_pass_to_pass} PASS_TO_PASS nodes or more than "
                f"{max_fail_to_pass} FAIL_TO_PASS nodes are dropped before selection, because the "
                "verifier runs every PASS_TO_PASS node on every evaluation"
            ),
            "changing_a_test_task_rule": (
                "final-test is chosen before the model is trained and is never re-drawn in "
                "response to a score, including a score of zero"
            ),
        },
        "requested_sizes": sizes,
        "realised_counts": {name: counts.get(name, 0) for name in sizes},
        "repositories_per_split": per_split_repos,
        "pool_stats": stats,
        "instances": instances,
    }


def render_summary(manifest: dict[str, Any]) -> str:
    lines = ["# route-v2 task split (provisional)", ""]
    for name in ("train", "development", "final-test"):
        subset = [item for item in manifest["instances"] if item["split"] == name]
        repos = Counter(item["repo"] for item in subset)
        ridable = sum(1 for item in subset if item["test_command_status"] == "audited")
        lines.append(
            f"- **{name}**: {len(subset)} instances, audited test command on {ridable}, "
            f"repos {dict(repos)}"
        )
        for item in subset:
            flag = "" if item["test_command_status"] == "audited" else " (unaudited)"
            lines.append(
                f"    - `{item['instance_id']}` {item['repo']}@{item['version']} "
                f"f2p={item['fail_to_pass_count']} p2p={item['pass_to_pass_count']}{flag}"
            )
    lines.append("")
    lines.append(f"pool: {manifest['pool_stats']}")
    return "\n".join(lines) + "\n"


def render_image_list(manifest: dict[str, Any], *, splits: tuple[str, ...]) -> str:
    """One image per line, for `scripts/vm/pull_task_images.sh` to pre-pull on the worker.

    Pre-pulling is mandatory rather than an optimisation: the worker's `registry-1.docker.io`
    is DNS-poisoned, so a container start cannot fetch a missing image, and a rollout would
    fail as an infrastructure error instead of a policy error.
    """

    wanted = set(splits)
    images = sorted(
        {item["image"] for item in manifest["instances"] if item["split"] in wanted}
    )
    return "\n".join(images) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, default=REPOSITORY_ROOT / "work/private/swe-gym-train-rows.parquet")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "manifests/route-v2-task-split.json")
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--image-list",
        dest="image_list",
        type=Path,
        help="write the images the train and final-test splits need, one per line",
    )
    parser.add_argument("--salt", default=DEFAULT_SALT)
    parser.add_argument("--train", type=int, default=12)
    parser.add_argument(
        "--final-test",
        dest="final_test",
        type=int,
        default=8,
        help="the number of clean final-test instances to draw",
    )
    parser.add_argument(
        "--min-development",
        dest="min_development",
        type=int,
        default=10,
        help=(
            "lower bound on the development set. Development is defined by the route-v1 rule, "
            "not by a quota, so this only guards against that rule silently matching nothing."
        ),
    )
    parser.add_argument(
        "--max-pass-to-pass",
        dest="max_pass_to_pass",
        type=int,
        default=80,
        help="drop instances with more PASS_TO_PASS nodes than this; the verifier runs them all",
    )
    parser.add_argument(
        "--max-fail-to-pass",
        dest="max_fail_to_pass",
        type=int,
        default=8,
        help="drop instances whose fix must turn on an implausibly large number of tests",
    )
    parser.add_argument(
        "--exclude-images",
        dest="exclude_images",
        type=Path,
        help=(
            "a file of image names known not to exist upstream (one per line), as produced by "
            "scripts/vm/check_images_exist.sh; every instance using one is dropped"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    from coding_agent_rl_lab.swe_gym import SWEGymAdapterError, audited_swe_gym_test_command

    # Ask the adapter itself which repo/version pairs it can run, so this script cannot drift
    # from the audited table it depends on.
    audited: set[tuple[str, str]] = set()
    for repo, version in DEFAULT_SOURCES:
        try:
            # The accessor validates the whole row, so give it the minimum it needs to answer
            # the only question being asked here: is this repo/version pair in the table?
            audited_swe_gym_test_command(
                {"repo": repo, "version": version, "FAIL_TO_PASS": ["probe"], "PASS_TO_PASS": []}
            )
        except SWEGymAdapterError:
            continue
        audited.add((repo, version))
    if not audited:
        print("FAILED  the adapter has no audited test command for any requested repository")
        return 1

    sizes = {
        "train": arguments.train,
        "development": arguments.min_development,
        "final-test": arguments.final_test,
    }
    rows = load_shard(arguments.shard, DEFAULT_SOURCES)
    rows, cost_dropped = apply_cost_filter(
        rows,
        max_pass_to_pass=arguments.max_pass_to_pass,
        max_fail_to_pass=arguments.max_fail_to_pass,
    )
    unavailable: set[str] = set()
    if arguments.exclude_images:
        unavailable = {
            line.strip()
            for line in arguments.exclude_images.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    rows, image_dropped = apply_image_availability_filter(rows, unavailable)
    allocations, stats = assign(rows, sizes, salt=arguments.salt)
    stats["cost_filter_dropped"] = cost_dropped
    stats["images_unavailable_dropped"] = image_dropped
    stats["pool_instances_after_filters"] = len(rows)
    problems = validate(allocations, rows, sizes=sizes)
    if problems:
        for problem in problems:
            print(f"FAILED  {problem}")
        return 1

    manifest = build_manifest(
        rows,
        allocations,
        stats,
        sizes=sizes,
        salt=arguments.salt,
        shard=arguments.shard,
        sources=DEFAULT_SOURCES,
        audited=audited,
        max_pass_to_pass=arguments.max_pass_to_pass,
        max_fail_to_pass=arguments.max_fail_to_pass,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    summary = render_summary(manifest)
    print(summary)
    if arguments.summary:
        arguments.summary.parent.mkdir(parents=True, exist_ok=True)
        arguments.summary.write_text(summary, encoding="utf-8")
        print(f"wrote {arguments.summary}")
    if arguments.image_list:
        images = render_image_list(manifest, splits=("development", "train", "final-test"))
        arguments.image_list.parent.mkdir(parents=True, exist_ok=True)
        arguments.image_list.write_text(images, encoding="utf-8")
        print(f"wrote {arguments.image_list} ({len(images.splitlines())} images)")
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
