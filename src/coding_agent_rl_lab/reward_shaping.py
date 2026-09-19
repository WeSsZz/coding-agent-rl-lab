from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Sequence

from .contracts import TestResult, VerifierBreakdown


_FAILED_NODE_RE = re.compile(r"^FAILED\s+([^\s]+)", re.MULTILINE)
_FAILED_COUNT_RE = re.compile(r"(\d+)\s+failed\b")
_COLLECTION_ERROR_RE = re.compile(r"\b[1-9]\d* errors?\b|ERROR collecting")
REWARD_VERSIONS = ("legacy-v1", "conservative-v2")


def verifier_failure_ids(result: TestResult | None) -> frozenset[str]:
    if result is None or result.passed:
        return frozenset()
    output = f"{result.stdout}\n{result.stderr}"
    return frozenset(_FAILED_NODE_RE.findall(output))


def verifier_failure_count(result: TestResult | None) -> int | None:
    if result is None:
        return None
    if result.passed:
        return 0
    failure_ids = verifier_failure_ids(result)
    if failure_ids:
        return len(failure_ids)
    counts = [int(value) for value in _FAILED_COUNT_RE.findall(f"{result.stdout}\n{result.stderr}")]
    return counts[-1] if counts else None


def verifier_collection_error(result: TestResult | None) -> bool:
    """True when the run failed while collecting tests, so no per-test result exists."""

    if result is None:
        return False
    return bool(_COLLECTION_ERROR_RE.search(f"{result.stdout}\n{result.stderr}"))


def _matches_target(node: str, target: str) -> bool:
    return node == target or node.startswith(f"{target}[")


def verifier_breakdown(
    result: TestResult | None,
    *,
    fail_to_pass: Sequence[str] = (),
    pass_to_pass: Sequence[str] = (),
) -> VerifierBreakdown | None:
    """Map one verifier run onto the declared FAIL_TO_PASS and PASS_TO_PASS nodes."""

    if result is None:
        return None
    fail_to_pass = tuple(fail_to_pass)
    pass_to_pass = tuple(pass_to_pass)
    declared = bool(fail_to_pass or pass_to_pass)
    if verifier_collection_error(result):
        return VerifierBreakdown(
            fail_to_pass_total=len(fail_to_pass),
            pass_to_pass_total=len(pass_to_pass),
            node_targets_declared=declared,
            collection_error=True,
        )
    failed_nodes = tuple(sorted(verifier_failure_ids(result)))
    if not declared:
        return VerifierBreakdown(failed_nodes=failed_nodes)
    if not result.passed and not failed_nodes:
        # A non-zero exit without a per-test summary cannot be charged to any node.
        return VerifierBreakdown(
            fail_to_pass_total=len(fail_to_pass),
            pass_to_pass_total=len(pass_to_pass),
            node_targets_declared=True,
        )
    graded_targets = (*fail_to_pass, *pass_to_pass)
    return VerifierBreakdown(
        fail_to_pass_total=len(fail_to_pass),
        pass_to_pass_total=len(pass_to_pass),
        fail_to_pass_resolved=sum(
            1
            for target in fail_to_pass
            if not any(_matches_target(node, target) for node in failed_nodes)
        ),
        pass_to_pass_regressed=sum(
            1
            for node in failed_nodes
            if any(_matches_target(node, target) for target in pass_to_pass)
        ),
        failed_nodes=failed_nodes,
        ungraded_failed_nodes=tuple(
            node
            for node in failed_nodes
            if not any(_matches_target(node, target) for target in graded_targets)
        ),
        node_targets_declared=True,
    )


@dataclass(frozen=True)
class TrainingReward:
    strict_success: bool
    baseline_failure_count: int | None
    final_failure_count: int | None
    resolved_failure_count: int
    new_failure_count: int | None
    patch_created: bool
    patch_valid: bool
    verifier_run_after_patch: bool
    violations: tuple[str, ...]
    strict_reward: float
    training_reward: float
    reward_version: str = "legacy-v1"
    verifier: VerifierBreakdown | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_training_reward(
    *,
    baseline: TestResult | None,
    final: TestResult | None,
    patch_created: bool,
    patch_valid: bool,
    verifier_run_after_patch: bool,
    violations: tuple[str, ...],
    reward_version: str = "legacy-v1",
    fail_to_pass: Sequence[str] = (),
    pass_to_pass: Sequence[str] = (),
) -> TrainingReward:
    if reward_version not in REWARD_VERSIONS:
        raise ValueError("unknown training reward version")
    strict_success = bool(
        final
        and final.passed
        and patch_created
        and not violations
    )
    baseline_ids = verifier_failure_ids(baseline)
    final_ids = verifier_failure_ids(final)
    baseline_count = verifier_failure_count(baseline)
    final_count = verifier_failure_count(final)
    # A collection error or unparseable verifier output is unknown, not zero
    # failures. Preserve a conservative audit count without inventing progress.
    resolved_count = (
        max(0, baseline_count - final_count)
        if baseline_count is not None and final_count is not None
        else 0
    )
    new_failure_count: int | None = None
    if baseline_ids and final is not None and (final.passed or final_ids):
        new_failure_count = len(final_ids - baseline_ids)

    if violations:
        training_reward = -1.0
    elif strict_success:
        training_reward = 1.0
    elif final is None or final.timed_out or not patch_created:
        training_reward = 0.0
    elif reward_version == "conservative-v2":
        # Only complete, comparable verifier failures can earn partial credit.
        # Unknown status, invalid patches and newly failing tests earn nothing.
        comparable = bool(
            baseline is not None and not baseline.timed_out
            and baseline.exit_code == 1 and final.exit_code == 1
            and baseline_count and final_count is not None
            and baseline_ids and final_ids
            and new_failure_count == 0
            and patch_valid and verifier_run_after_patch
            and not verifier_collection_error(final)
        )
        training_reward = (
            round(0.03 + (0.25 * resolved_count / baseline_count + 0.10 if resolved_count else 0.0), 4)
            if comparable else 0.0
        )
    else:
        reward = 0.0
        if baseline_count and final_count is not None and final_count < baseline_count:
            reward += 0.25 * resolved_count / baseline_count
            if new_failure_count == 0:
                reward += 0.10
        if patch_valid and verifier_run_after_patch:
            reward += 0.10
            reward += 0.05
        training_reward = round(min(0.50, reward), 4)

    return TrainingReward(
        strict_success=strict_success,
        baseline_failure_count=baseline_count,
        final_failure_count=final_count,
        resolved_failure_count=resolved_count,
        new_failure_count=new_failure_count,
        patch_created=patch_created,
        patch_valid=patch_valid,
        verifier_run_after_patch=verifier_run_after_patch,
        violations=violations,
        strict_reward=float(strict_success),
        training_reward=training_reward,
        reward_version=reward_version,
        verifier=verifier_breakdown(
            final,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
        ),
    )
