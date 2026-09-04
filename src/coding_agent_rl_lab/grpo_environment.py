from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

from .contracts import ActionKind, AgentAction, CodingTask, TestResult
from .environment import CodingEnvironment, EnvironmentError
from .providers import EnvironmentProvider


class GRPOEnvironmentError(RuntimeError):
    pass


def build_grpo_prompt_rows(tasks: Iterable[CodingTask]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "task_id": task.task_id,
            "prompt": [
                {"role": "system", "content": _GRPO_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task_id": task.task_id,
                            "issue": task.issue,
                            "base_commit": task.base_commit,
                            "repository": {
                                key: task.metadata[key]
                                for key in ("repo", "version")
                                if key in task.metadata
                            },
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        for task in tasks
    )


def build_grpo_environment_factory(
    tasks: Iterable[CodingTask],
    provider: EnvironmentProvider,
) -> Callable[[], GRPOCodingEnvironment]:
    task_map = {task.task_id: task for task in tasks}
    if not task_map:
        raise ValueError("GRPO task set must not be empty")

    def factory() -> GRPOCodingEnvironment:
        return GRPOCodingEnvironment(task_map, provider)

    return factory


class GRPOCodingEnvironment:
    """TRL-compatible stateful tool environment backed by the audited provider."""

    def __init__(self, tasks: dict[str, CodingTask], provider: EnvironmentProvider) -> None:
        self._tasks = dict(tasks)
        self._provider = provider
        self._environment: CodingEnvironment | None = None
        self._completed = False
        self._reward = 0.0

    def reset(self, **kwargs: Any) -> str:
        task_id = kwargs.get("task_id")
        if not isinstance(task_id, str) or task_id not in self._tasks:
            raise GRPOEnvironmentError("reset requires a known task_id")
        self._close_environment()
        task = self._tasks[task_id]
        self._environment = self._provider.create(task)
        self._completed = False
        self._reward = 0.0
        return self._environment.reset(task)

    def list_files(self) -> str:
        """List repository files visible to the coding agent.

        Returns:
            A bounded newline-delimited repository file listing.
        """

        return self._act(AgentAction(ActionKind.LIST_FILES))

    def search_text(self, query: str) -> str:
        """Search repository file paths and literal text.

        Args:
            query: A non-empty literal identifier, text, or filename fragment.

        Returns:
            Bounded matching paths, line numbers, and text.
        """

        return self._act(AgentAction(ActionKind.SEARCH_TEXT, {"query": query}))

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """Read a repository file using a relative path.

        Args:
            path: Repository-relative path previously observed in tool output.
            start_line: Optional one-based first line; provide with end_line.
            end_line: Optional inclusive last line; provide with start_line.

        Returns:
            Bounded file contents.
        """

        arguments: dict[str, Any] = {"path": path}
        if start_line is not None or end_line is not None:
            arguments.update({"start_line": start_line, "end_line": end_line})
        return self._act(AgentAction(ActionKind.READ_FILE, arguments))

    def replace_text(self, path: str, old: str, new: str) -> str:
        """Replace one exact text occurrence in a non-test repository file.

        Args:
            path: Repository-relative source file path.
            old: Exact non-empty text to replace once.
            new: Replacement text.

        Returns:
            The environment update result.
        """

        return self._act(
            AgentAction(
                ActionKind.REPLACE_TEXT,
                {"path": path, "old": old, "new": new},
            )
        )

    def replace_lines(self, path: str, start_line: int, end_line: int, new: str) -> str:
        """Replace a small line range in a source file that was already read.

        Args:
            path: Repository-relative source file path.
            start_line: One-based first line to replace.
            end_line: Inclusive final line to replace, at most 80 lines total.
            new: Replacement text.

        Returns:
            The environment update result.
        """

        return self._act(
            AgentAction(
                ActionKind.REPLACE_LINES,
                {
                    "path": path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "new": new,
                },
            )
        )

    def run_tests(self) -> str:
        """Run the fixed verifier command after a code change.

        Returns:
            The bounded verifier output.
        """

        return self._act(AgentAction(ActionKind.RUN_TESTS))

    def finish(self) -> str:
        """Finish the episode and score the final repository state.

        Returns:
            The final verifier output.
        """

        return self._act(AgentAction(ActionKind.FINISH))

    @property
    def reward(self) -> float:
        if self._environment is None:
            return self._reward
        if not self._completed:
            self._finalize()
        return self._reward

    def get_reward(self) -> float:
        """Return the verifier reward expected by current TRL environments.

        Returns:
            One for a verified non-empty source patch, otherwise zero.
        """

        return self.reward

    def _act(self, action: AgentAction) -> str:
        environment = self._require_active()
        try:
            result = environment.step(action)
        except EnvironmentError as exc:
            self._reward = 0.0
            self._completed = True
            self._close_environment()
            raise GRPOEnvironmentError(str(exc)) from exc
        if result.terminated:
            self._complete(result.test_result)
        return result.observation

    def _finalize(self) -> None:
        environment = self._require_active()
        self._complete(environment.finalize())

    def _complete(self, test_result: TestResult | None) -> None:
        environment = self._require_active()
        changed_files = environment.changed_files()
        passed = bool(test_result and test_result.passed)
        self._reward = float(passed and bool(changed_files) and not environment.violations)
        self._completed = True
        self._close_environment()

    def _require_active(self) -> CodingEnvironment:
        if self._environment is None or self._completed:
            raise GRPOEnvironmentError("environment is not active; call reset with a task_id")
        return self._environment

    def _close_environment(self) -> None:
        if self._environment is not None:
            self._environment.close()
            self._environment = None


def grpo_verifier_reward(environments: Iterable[GRPOCodingEnvironment], **_: Any) -> list[float]:
    return [environment.reward for environment in environments]


_GRPO_SYSTEM_PROMPT = """You are a coding agent in a restricted repository environment.
Use the provided tools to inspect the failing behavior, make the smallest relevant source change,
and run the verifier. Never modify tests or escape the repository. Stop only after using finish.
Tool errors are observations: change strategy instead of repeating an unchanged action.
Prefer replace_text for a known exact snippet. If exact matching fails, use replace_lines only on a
small one-based line range from a file you already read, then run the verifier.

Every assistant turn must contain exactly one tool call and no prose. Use this exact format:
<tool_call>
{"name":"search_text","arguments":{"query":"literal identifier"}}
</tool_call>
Replace the example name and arguments with the selected provided tool. Never use Markdown fences,
never describe a future tool call, and never answer with a plain JSON action object.
"""
