from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .contracts import TestResult


_FAILED_NODE_RE = re.compile(r"^FAILED\s+([^\s]+)", re.MULTILINE)
_FAILED_COUNT_RE = re.compile(r"(\d+)\s+failed\b")


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
) -> TrainingReward:
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
    resolved_count = max(0, (baseline_count or 0) - (final_count or 0))
    new_failure_count: int | None = None
    if baseline_ids and final is not None:
        new_failure_count = len(final_ids - baseline_ids)

    if violations:
        training_reward = -1.0
    elif strict_success:
        training_reward = 1.0
    elif final is None or final.timed_out or not patch_created:
        training_reward = 0.0
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
    )
