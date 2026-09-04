from __future__ import annotations

import argparse
import json
import os
import threading
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .contracts import ActionKind, AgentAction, CodingTask, TestResult
from .docker_environment import DockerSandboxConfig
from .environment import CodingEnvironment, EnvironmentError
from .evaluation import load_builtin_tasks
from .grpo_environment import GRPOEnvironmentError
from .providers import DockerSandboxProvider, EnvironmentProvider, LocalFixtureEnvironmentProvider
from .reward_shaping import TrainingReward, build_training_reward
from .swe_gym import SWEGymTaskAdapter, audited_swe_gym_test_command
from .swe_gym_smoke import (
    PINNED_INSTANCE_IDS,
    SWE_GYM_TASK_SET_CHOICES,
    dataset_split_for_task_set,
    load_or_download_pinned_rows,
    pinned_rows_for_task_set,
)


_REWARD_AUDIT_LOCK = threading.Lock()


class GRPOWorkerError(RuntimeError):
    pass


@dataclass
class _WorkerSession:
    environment: CodingEnvironment | None
    action_kinds: list[str] = field(default_factory=list)
    reward: float = 0.0
    strict_reward: float = 0.0
    reward_components: TrainingReward | None = None
    verifier_run_after_patch: bool = False
    completed: bool = False


class GRPOWorker:
    def __init__(self, tasks: dict[str, CodingTask], provider: EnvironmentProvider) -> None:
        if not tasks:
            raise ValueError("worker task set must not be empty")
        self._tasks = dict(tasks)
        self._provider = provider
        self._sessions: dict[str, _WorkerSession] = {}
        self._lock = threading.Lock()

    def create(self, task_id: str) -> dict[str, Any]:
        try:
            task = self._tasks[task_id]
        except KeyError as exc:
            raise GRPOWorkerError("unknown task_id") from exc
        environment = self._provider.create(task)
        try:
            observation = environment.reset(task)
        except Exception:
            environment.close()
            raise
        session_id = uuid.uuid4().hex
        with self._lock:
            self._sessions[session_id] = _WorkerSession(environment=environment)
        return {"session_id": session_id, "observation": observation}

    def action(self, session_id: str, action: AgentAction) -> dict[str, Any]:
        session = self._session(session_id)
        environment = self._active_environment(session)
        session.action_kinds.append(action.kind.value)
        patch_existed_before_action = bool(environment.changed_files())
        try:
            result = environment.step(action)
        except EnvironmentError as exc:
            self._complete(session, None)
            raise GRPOWorkerError(str(exc)) from exc
        if (
            patch_existed_before_action
            and action.kind in {ActionKind.RUN_TESTS, ActionKind.FINISH}
        ):
            session.verifier_run_after_patch = True
        if result.terminated:
            self._complete(session, result.test_result)
        return {
            "observation": result.observation,
            "terminated": session.completed,
            "reward": session.reward if session.completed else None,
            "strict_reward": session.strict_reward if session.completed else None,
            "reward_components": (
                session.reward_components.to_dict()
                if session.completed and session.reward_components is not None
                else None
            ),
            "action_kinds": list(session.action_kinds) if session.completed else None,
        }

    def finalize(self, session_id: str) -> dict[str, Any]:
        session = self._session(session_id)
        if not session.completed:
            environment = self._active_environment(session)
            self._complete(session, environment.finalize())
        return {
            "reward": session.reward,
            "strict_reward": session.strict_reward,
            "reward_components": (
                session.reward_components.to_dict()
                if session.reward_components is not None
                else None
            ),
            "action_kinds": list(session.action_kinds),
        }

    def delete(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session and session.environment is not None:
            session.environment.close()
        return {"deleted": session is not None}

    def close(self) -> None:
        with self._lock:
            session_ids = tuple(self._sessions)
        for session_id in session_ids:
            self.delete(session_id)

    def _session(self, session_id: str) -> _WorkerSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise GRPOWorkerError("unknown session")
        return session

    @staticmethod
    def _active_environment(session: _WorkerSession) -> CodingEnvironment:
        if session.completed or session.environment is None:
            raise GRPOWorkerError("session is already complete")
        return session.environment

    @staticmethod
    def _complete(session: _WorkerSession, test_result: TestResult | None) -> None:
        environment = session.environment
        if environment is None:
            return
        try:
            changed_files = environment.changed_files()
            reward = build_training_reward(
                baseline=environment.baseline_result,
                final=test_result,
                patch_created=bool(changed_files),
                patch_valid=environment.patch_is_valid(),
                verifier_run_after_patch=session.verifier_run_after_patch,
                violations=tuple(environment.violations),
            )
            session.reward = reward.training_reward
            session.strict_reward = reward.strict_reward
            session.reward_components = reward
            session.completed = True
        finally:
            environment.close()
            session.environment = None


def build_worker_server(
    worker: GRPOWorker,
    *,
    token: str,
    port: int = 9010,
) -> ThreadingHTTPServer:
    if len(token) < 32:
        raise ValueError("worker token must contain at least 32 characters")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if not self._authorized():
                self._write(401, {"error": "unauthorized"})
                return
            try:
                body = self._json_body()
                if self.path == "/v1/sessions":
                    result = worker.create(str(body.get("task_id", "")))
                elif self.path.endswith("/actions"):
                    session_id = self.path.removeprefix("/v1/sessions/").removesuffix("/actions")
                    action_payload = body.get("action")
                    if not isinstance(action_payload, dict):
                        raise GRPOWorkerError("action must be an object")
                    result = worker.action(session_id, AgentAction.from_dict(action_payload))
                elif self.path.endswith("/finalize"):
                    session_id = self.path.removeprefix("/v1/sessions/").removesuffix("/finalize")
                    result = worker.finalize(session_id)
                else:
                    self._write(404, {"error": "not found"})
                    return
                self._write(200, result)
            except (GRPOWorkerError, KeyError, TypeError, ValueError) as exc:
                self._write(409, {"error": str(exc)})

        def do_DELETE(self) -> None:
            if not self._authorized():
                self._write(401, {"error": "unauthorized"})
                return
            prefix = "/v1/sessions/"
            if not self.path.startswith(prefix):
                self._write(404, {"error": "not found"})
                return
            self._write(200, worker.delete(self.path[len(prefix) :]))

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

        def _authorized(self) -> bool:
            return self.headers.get("Authorization") == f"Bearer {token}"

        def _json_body(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise GRPOWorkerError("invalid content length") from exc
            if length <= 0 or length > 1_000_000:
                raise GRPOWorkerError("request body size is invalid")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise GRPOWorkerError("request body must be an object")
            return payload

        def _write(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    original_close = server.server_close

    def close() -> None:
        worker.close()
        original_close()

    server.server_close = close  # type: ignore[method-assign]
    return server


class RemoteGRPOCodingEnvironment:
    """TRL tool environment client for the loopback-only Ubuntu worker API."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        reward_audit_path: Path | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._reward_audit_path = reward_audit_path
        self._session_id: str | None = None
        self._task_id: str | None = None
        self._reward = 0.0
        self._completed = False

    def reset(self, **kwargs: Any) -> str:
        task_id = kwargs.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise GRPOEnvironmentError("reset requires task_id")
        self._delete()
        payload = self._request("POST", "/v1/sessions", {"task_id": task_id})
        self._session_id = str(payload["session_id"])
        self._task_id = task_id
        self._reward = 0.0
        self._completed = False
        return str(payload["observation"])

    def list_files(self) -> str:
        """List repository files.

        Returns:
            A bounded repository file listing.
        """
        return self._act(AgentAction(ActionKind.LIST_FILES))

    def search_text(self, query: str) -> str:
        """Search repository file paths and literal text.

        Args:
            query: Literal identifier, text, or filename fragment.

        Returns:
            Bounded repository matches.
        """
        return self._act(AgentAction(ActionKind.SEARCH_TEXT, {"query": query}))

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """Read a repository file.

        Args:
            path: Repository-relative file path.
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
        """Replace one exact source text occurrence.

        Args:
            path: Repository-relative source file path.
            old: Exact non-empty text to replace.
            new: Replacement text.

        Returns:
            The update result.
        """
        return self._act(AgentAction(ActionKind.REPLACE_TEXT, {"path": path, "old": old, "new": new}))

    def run_tests(self) -> str:
        """Run the fixed verifier.

        Returns:
            Bounded verifier output.
        """
        return self._act(AgentAction(ActionKind.RUN_TESTS))

    def finish(self) -> str:
        """Finish and score the repository state.

        Returns:
            Final verifier output.
        """
        return self._act(AgentAction(ActionKind.FINISH))

    @property
    def reward(self) -> float:
        if self._session_id is None:
            return self._reward
        if not self._completed:
            session_id = self._require_session()
            payload = self._request("POST", f"/v1/sessions/{session_id}/finalize", {})
            self._reward = float(payload["reward"])
            self._completed = True
            self._audit_reward(payload, completion_source="finalize")
            self._delete()
        return self._reward

    def get_reward(self) -> float:
        """Return the deterministic shaped training reward expected by TRL.

        Returns:
            One for strict verifier success, a bounded partial reward for
            verified progress, zero for no progress, or negative one for a
            safety violation.
        """

        return self.reward

    def _act(self, action: AgentAction) -> str:
        session_id = self._require_session()
        try:
            payload = self._request(
                "POST",
                f"/v1/sessions/{session_id}/actions",
                {"action": action.to_dict()},
            )
        except GRPOEnvironmentError:
            self._completed = True
            self._reward = 0.0
            self._delete()
            raise
        if payload.get("terminated"):
            self._completed = True
            self._reward = float(payload.get("reward") or 0.0)
            self._audit_reward(payload, completion_source="action")
            self._delete()
        return str(payload["observation"])

    def _audit_reward(self, payload: dict[str, Any], *, completion_source: str) -> None:
        if self._reward_audit_path is None:
            return
        components = payload.get("reward_components")
        if not isinstance(components, dict):
            components = None
        record = {
            "schema_version": 1,
            "task_id": self._task_id,
            "completion_source": completion_source,
            "reward": _optional_float(payload.get("reward")),
            "strict_reward": _optional_float(payload.get("strict_reward")),
            "reward_components": components,
            "action_kinds": _action_kind_list(payload.get("action_kinds")),
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with _REWARD_AUDIT_LOCK:
            self._reward_audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self._reward_audit_path.open("a", encoding="utf-8") as stream:
                stream.write(line)

    def _request(self, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                decoded = json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            finally:
                exc.close()
            raise GRPOEnvironmentError(f"worker rejected request: {detail[-2000:]}") from exc
        except (OSError, ValueError) as exc:
            raise GRPOEnvironmentError(f"worker request failed: {exc}") from exc
        if not isinstance(decoded, dict):
            raise GRPOEnvironmentError("worker response must be an object")
        return decoded

    def _require_session(self) -> str:
        if self._session_id is None or self._completed:
            raise GRPOEnvironmentError("remote environment is not active")
        return self._session_id

    def _delete(self) -> None:
        if self._session_id is None:
            return
        session_id = self._session_id
        self._session_id = None
        try:
            self._request("DELETE", f"/v1/sessions/{session_id}", {})
        except GRPOEnvironmentError:
            pass


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _action_kind_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return None
    allowed = {kind.value for kind in ActionKind}
    if any(item not in allowed for item in value):
        return None
    return list(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the loopback-only GRPO Docker worker")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument("--task-count", type=int, default=None)
    parser.add_argument(
        "--task-source",
        choices=("swe-gym", "fixtures"),
        default="swe-gym",
    )
    parser.add_argument("--task-set", choices=SWE_GYM_TASK_SET_CHOICES, default="all")
    parser.add_argument(
        "--task-id",
        action="append",
        choices=PINNED_INSTANCE_IDS,
        default=[],
        help="Exact pinned SWE-Gym instance id to serve; repeat for multiple tasks.",
    )
    parser.add_argument("--rows-cache", default="work/swe-gym-development-rows.jsonl")
    parser.add_argument("--test-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--token-file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.task_count is not None and args.task_count <= 0:
        raise SystemExit("--task-count must be positive")
    if args.task_id and args.task_count is not None:
        raise SystemExit("--task-id and --task-count cannot be used together")
    if args.test_timeout_seconds <= 0:
        raise SystemExit("--test-timeout-seconds must be positive")
    token = os.environ.get("CODING_AGENT_GRPO_WORKER_TOKEN", "")
    if not token and args.token_file:
        token = Path(args.token_file).read_text(encoding="utf-8").strip()
    project_root = Path(__file__).resolve().parents[2]
    if args.task_source == "fixtures":
        if args.task_id:
            raise SystemExit("--task-id is only supported with --task-source swe-gym")
        available_tasks = load_builtin_tasks(project_root)
        task_count = args.task_count if args.task_count is not None else min(3, len(available_tasks))
        if task_count > len(available_tasks):
            raise SystemExit(f"--task-count cannot exceed {len(available_tasks)} for fixtures")
        tasks = available_tasks[:task_count]
        provider: EnvironmentProvider = LocalFixtureEnvironmentProvider(project_root)
    else:
        available = pinned_rows_for_task_set(args.task_set)
        task_count = args.task_count if args.task_count is not None else min(3, len(available))
        if not args.task_id and task_count > len(available):
            raise SystemExit(
                f"--task-count cannot exceed {len(available)} for task set {args.task_set}"
            )
        try:
            rows = load_or_download_pinned_rows(
                project_root / args.rows_cache,
                limit=None if args.task_id else task_count,
                task_set=args.task_set,
                task_ids=args.task_id,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        split = dataset_split_for_task_set(args.task_set)
        adapter = SWEGymTaskAdapter()
        bundles = tuple(
            adapter.adapt(
                row,
                split=split,
                test_command=audited_swe_gym_test_command(row),
            )
            for row in rows
        )
        tasks = tuple(bundle.task for bundle in bundles)
        provider = DockerSandboxProvider(
            {bundle.task.task_id: bundle.environment for bundle in bundles},
            DockerSandboxConfig(
                memory_limit="4g",
                cpu_limit=2.0,
                pids_limit=512,
                startup_timeout_seconds=180.0,
                command_timeout_seconds=120.0,
                test_timeout_seconds=args.test_timeout_seconds,
            ),
        )
    worker = GRPOWorker({task.task_id: task for task in tasks}, provider)
    server = build_worker_server(worker, token=token, port=args.port)
    print(
        f"GRPO worker listening on 127.0.0.1:{args.port} "
        f"source={args.task_source} task_set={args.task_set} tasks={len(tasks)}",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
