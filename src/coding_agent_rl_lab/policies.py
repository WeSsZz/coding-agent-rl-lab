from __future__ import annotations

from typing import Protocol, Sequence

from .contracts import ActionKind, AgentAction, CodingTask, PolicyManifest, TrajectoryStep


class Policy(Protocol):
    manifest: PolicyManifest

    def next_action(self, task: CodingTask, history: Sequence[TrajectoryStep]) -> AgentAction: ...


class NoOpPolicy:
    manifest = PolicyManifest(
        policy_id="noop",
        version="1",
        policy_type="deterministic_baseline",
        metadata={"purpose": "expected-failure control"},
    )

    def next_action(self, task: CodingTask, history: Sequence[TrajectoryStep]) -> AgentAction:
        del task
        return AgentAction(ActionKind.RUN_TESTS if not history else ActionKind.FINISH)


class ReferencePolicy:
    """Answer-bearing infrastructure check; never report this as a learned baseline."""

    manifest = PolicyManifest(
        policy_id="reference",
        version="1",
        policy_type="scripted_reference",
        metadata={"contains_answers": True, "purpose": "pipeline verification only"},
    )

    def __init__(self, actions: dict[str, tuple[AgentAction, ...]]) -> None:
        self.actions = actions

    def next_action(self, task: CodingTask, history: Sequence[TrajectoryStep]) -> AgentAction:
        sequence = self.actions.get(task.task_id)
        if sequence is None:
            raise KeyError(f"no reference actions for task {task.task_id}")
        if len(history) >= len(sequence):
            return AgentAction(ActionKind.FINISH)
        return sequence[len(history)]

