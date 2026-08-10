from __future__ import annotations

from typing import Protocol, Sequence

from .contracts import ActionKind, AgentAction, CodingTask, PolicyManifest, TrajectoryStep
from .model import StructuredActionModel


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


class ModelCodingPolicy:
    SYSTEM_PROMPT = """You are a repository repair agent. Choose exactly one tool action.
Return only a JSON object with this schema:
{"kind":"list_files|read_file|replace_text|run_tests|finish","arguments":{}}
read_file arguments: {"path":"relative/path"}
replace_text arguments: {"path":"relative/path","old":"exact text","new":"replacement"}
Never access paths outside the repository. Inspect evidence before editing. Run tests after editing.
"""

    def __init__(self, model: StructuredActionModel) -> None:
        self.model = model
        self.manifest = PolicyManifest(
            policy_id="openai-compatible-coding-agent",
            version="1",
            policy_type="model_tool_loop",
            model=model.model_name,
            metadata={"action_contract": "coding-action-v1"},
        )

    def next_action(self, task: CodingTask, history: Sequence[TrajectoryStep]) -> AgentAction:
        recent = history[-6:]
        transcript = "\n\n".join(
            f"Step {step.sequence}\nAction: {step.action.to_dict()}\nObservation:\n{step.observation[-6000:]}"
            for step in recent
        )
        user_prompt = f"Issue:\n{task.issue}\n\nRecent trajectory:\n{transcript or '(none)'}\n\nChoose the next action."
        return self.model.complete_action(system_prompt=self.SYSTEM_PROMPT, user_prompt=user_prompt)
