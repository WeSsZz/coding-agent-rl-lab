from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TASK_ID = "getmoto__moto-7607"
CALLBACK_PATH = (
    "moto/stepfunctions/parser/asl/component/state/exec/state_task/service/"
    "state_task_service_callback.py"
)
FAIL_TO_PASS = (
    "tests/test_stepfunctions/parser/test_stepfunctions_dynamodb_integration.py::"
    "test_state_machine_calling_dynamodb_put_wait_for_task_token"
)
PASS_TO_PASS = (
    "tests/test_stepfunctions/parser/test_stepfunctions_dynamodb_integration.py::"
    "test_state_machine_calling_dynamodb_put_and_delete",
    "tests/test_stepfunctions/parser/test_stepfunctions_dynamodb_integration.py::"
    "test_state_machine_calling_dynamodb_put",
)


class AuditError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit saved moto-7607 rollout sessions")
    parser.add_argument("--source", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    report_path = Path(args.report)
    for path in (output, report_path):
        if path.exists():
            raise AuditError(f"refusing to overwrite existing output: {path}")

    sessions: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []
    for item in args.source:
        if "=" not in item:
            raise AuditError("--source must have LABEL=DIR form")
        label, raw_dir = item.split("=", 1)
        source_dir = Path(raw_dir)
        trace_path = source_dir / "tool-traces.jsonl"
        audit_path = source_dir / f"{TASK_ID}-reward-audit.jsonl"
        run_report_path = source_dir / "report.json"
        run_sessions = audit_source(label, trace_path, audit_path)
        sessions.extend(run_sessions)
        source_manifest.append(
            {
                "label": label,
                "session_count": len(run_sessions),
                "trace_path": str(trace_path),
                "trace_sha256": _sha256(trace_path),
                "reward_audit_path": str(audit_path),
                "reward_audit_sha256": _sha256(audit_path),
                "report_path": str(run_report_path),
                "report_sha256": _sha256(run_report_path),
            }
        )

    counts = Counter(row["primary_blocker"] for row in sessions)
    audit = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "session_count": len(sessions),
        "incomplete_session_count": sum(not row["audit_mapping_complete"] for row in sessions),
        "primary_blocker_counts": dict(sorted(counts.items())),
        "callback_reached_count": sum(row["read_callback"] for row in sessions),
        "full_target_function_read_count": sum(row["read_full_target_function"] for row in sessions),
        "strict_success_count": sum(row["strict_success"] for row in sessions),
        "failure_reduction_count": sum(row["resolved_failure_count"] > 0 for row in sessions),
        "new_failure_session_count": sum(row["new_failure_count"] > 0 for row in sessions),
        "sources": source_manifest,
        "sessions": sessions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_markdown_report(audit), encoding="utf-8")
    print(json.dumps({key: audit[key] for key in audit if key not in {"sessions", "sources"}}, indent=2))


def audit_source(label: str, trace_path: Path, audit_path: Path) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _jsonl(trace_path):
        request_path = str(row.get("path", ""))
        action = row.get("request", {}).get("action")
        if not action or "/sessions/" not in request_path or not request_path.endswith("/actions"):
            continue
        session_id = request_path.split("/sessions/", 1)[1].split("/", 1)[0]
        grouped[session_id].append(row)
    audits = _jsonl(audit_path)
    available = set(range(len(audits)))
    result: list[dict[str, Any]] = []
    for session_id, steps in grouped.items():
        kinds = [step["request"]["action"]["kind"] for step in steps]
        outcomes = [_outcome(step) for step in steps]
        candidates = [
            index
            for index in available
            if audits[index].get("action_kinds") == kinds
            and audits[index].get("action_outcomes") == outcomes
        ]
        if len(candidates) != 1:
            raise AuditError(
                f"{label}/{session_id}: audit mapping is not unique ({candidates!r})"
            )
        audit_index = candidates[0]
        available.remove(audit_index)
        result.append(_session_record(label, session_id, steps, audits[audit_index], audit_index))
    if available:
        raise AuditError(f"{label}: {len(available)} reward audits were not mapped to sessions")
    return result


def _session_record(
    label: str,
    session_id: str,
    steps: list[dict[str, Any]],
    audit: dict[str, Any],
    audit_index: int,
) -> dict[str, Any]:
    actions = [step["request"]["action"] for step in steps]
    observations = [str(step["response"].get("observation", "")) for step in steps]
    read_callback = any(
        action["kind"] == "read_file" and action.get("arguments", {}).get("path") == CALLBACK_PATH
        and not observation.startswith("Tool error:")
        for action, observation in zip(actions, observations, strict=True)
    )
    read_full_function = any(
        action["kind"] == "read_file"
        and action.get("arguments", {}).get("path") == CALLBACK_PATH
        and "def _wait_for_task_token(" in observation
        and "callback_endpoint = env.callback_pool_manager.get(callback_id)" in observation
        for action, observation in zip(actions, observations, strict=True)
    )
    edit_indexes = [
        index for index, action in enumerate(actions) if action["kind"] in {"replace_text", "replace_lines"}
    ]
    successful_edit_indexes = [
        index for index in edit_indexes if observations[index].startswith("Updated ")
    ]
    first_edit = None
    if edit_indexes:
        index = edit_indexes[0]
        first_edit = {
            "action_index": index,
            "kind": actions[index]["kind"],
            "path": actions[index].get("arguments", {}).get("path"),
            "outcome": _outcome(steps[index])["outcome"],
        }
    tool_errors = [
        {
            "action_index": index,
            "kind": actions[index]["kind"],
            "path": actions[index].get("arguments", {}).get("path"),
            "message": observation.splitlines()[0][:500],
        }
        for index, observation in enumerate(observations)
        if observation.startswith("Tool error:")
    ]
    modifications = [
        {
            "action_index": index,
            "kind": actions[index]["kind"],
            "arguments": actions[index]["arguments"],
        }
        for index in successful_edit_indexes
    ]
    components = audit.get("reward_components") or {}
    last_run_index = max(
        (index for index, action in enumerate(actions) if action["kind"] == "run_tests"),
        default=None,
    )
    last_edit_index = max(successful_edit_indexes, default=None)
    final_ids: list[str] | None = None
    final_ids_source = "not_recorded"
    verifier_error_types: list[str] = []
    observed_verifier_error_types = sorted(
        set(
            error
            for index, observation in enumerate(observations)
            if actions[index]["kind"] == "run_tests"
            for error in re.findall(
                r"\b(?:SyntaxError|IndentationError|ImportError|NameError|AttributeError)\b",
                observation,
            )
        )
    )
    if last_run_index is not None and (last_edit_index is None or last_run_index > last_edit_index):
        final_ids = sorted(set(re.findall(r"(?m)^FAILED\s+([^\s]+)", observations[last_run_index])))
        final_ids_source = "last_run_tests_trace"
        verifier_error_types = sorted(
            set(re.findall(r"\b(?:SyntaxError|IndentationError|ImportError|NameError|AttributeError)\b", observations[last_run_index]))
        )
    elif last_run_index is not None:
        final_ids_source = "finalizer_after_unverified_edit_not_recorded"

    if not read_callback:
        primary_blocker = "navigation_not_reached"
        gold_difference = "Never read the callback implementation; edits, when present, targeted unrelated modules."
    elif not successful_edit_indexes:
        primary_blocker = "insufficient_reading"
        gold_difference = "Read the callback implementation but produced no executable edit to its token-wait behavior."
    elif any(
        actions[index].get("arguments", {}).get("path") != CALLBACK_PATH
        for index in successful_edit_indexes
    ):
        primary_blocker = "semantic_edit_error"
        gold_difference = "Reached the callback implementation, then edited a different module instead of its missing-token wait path."
    elif tool_errors and not successful_edit_indexes:
        primary_blocker = "edit_execution_error"
        gold_difference = "Attempted the target edit but no edit executed successfully."
    else:
        primary_blocker = "unverified_or_unfinished"
        gold_difference = "Edited the target area but did not complete a verified correct recovery."

    return {
        "source": label,
        "task_id": TASK_ID,
        "session_id": session_id,
        "audit_row_index": audit_index,
        "audit_mapping_complete": True,
        "action_count": len(actions),
        "parseable_tool_call_count": len(actions),
        "action_kinds": [action["kind"] for action in actions],
        "read_callback": read_callback,
        "read_full_target_function": read_full_function,
        "first_edit": first_edit,
        "tool_errors": tool_errors,
        "successful_modifications": modifications,
        "baseline_failed_test_ids": [FAIL_TO_PASS],
        "final_failed_test_ids": final_ids,
        "final_failed_test_ids_source": final_ids_source,
        "regression_test_ids": (
            [test_id for test_id in PASS_TO_PASS if final_ids and test_id in final_ids]
            if final_ids is not None
            else None
        ),
        "verifier_error_types": verifier_error_types,
        "observed_verifier_error_types": observed_verifier_error_types,
        "verifier_run_after_patch": bool(components.get("verifier_run_after_patch")),
        "baseline_failure_count": components.get("baseline_failure_count"),
        "final_failure_count": components.get("final_failure_count"),
        "resolved_failure_count": components.get("resolved_failure_count", 0),
        "new_failure_count": components.get("new_failure_count", 0),
        "patch_valid": components.get("patch_valid"),
        "strict_success": bool(components.get("strict_success")),
        "reward": audit.get("reward"),
        "primary_blocker": primary_blocker,
        "gold_behavior_difference": gold_difference,
    }


def _outcome(step: dict[str, Any]) -> dict[str, str]:
    action = step["request"]["action"]
    kind = action["kind"]
    response = step["response"]
    observation = str(response.get("observation", ""))
    if observation.startswith("Tool error:"):
        outcome = "tool_error"
    elif kind in {"replace_text", "replace_lines"} and observation.startswith("Updated "):
        outcome = "updated"
    elif response.get("terminated"):
        outcome = "terminated"
    else:
        outcome = "ok"
    return {"kind": kind, "outcome": outcome}


def _markdown_report(audit: dict[str, Any]) -> str:
    counts = audit["primary_blocker_counts"]
    latest = [row for row in audit["sessions"] if row["source"] == "seed121100"]
    latest_regressions = [row for row in latest if row["new_failure_count"] > 0]
    error_summary = Counter(
        error
        for row in latest_regressions
        for error in row["observed_verifier_error_types"]
    )
    lines = [
        "# moto-7607 阶段 A：真实 rollout 归因",
        "",
        f"共重建 {audit['session_count']} 个独立 session；trace 与 reward audit 均按完整 action_kinds + action_outcomes 唯一映射，未发现 incomplete session。",
        "",
        "## 主要结论",
        "",
        f"- 首个阻断点：{json.dumps(counts, ensure_ascii=False, sort_keys=True)}。",
        f"- 读到 callback 文件：{audit['callback_reached_count']}/{audit['session_count']}；读到完整 `_wait_for_task_token`：{audit['full_target_function_read_count']}/{audit['session_count']}。",
        f"- 完整修复：{audit['strict_success_count']}/{audit['session_count']}；failure reduction：{audit['failure_reduction_count']}/{audit['session_count']}。",
        "- gold 行为差异不是文本距离问题：正确路径需要在 callback 的 token-wait 分支处理缺失 Task/Token，并运行完整 verifier；绝大多数 rollout 没有到达该实现，少数到达后仍编辑了 AWS SDK 父实现。",
        "",
        "## 最新 seed121100 的 1→3 failures",
        "",
        f"3 个回归 session 均有无关模块的宽范围替换，patch_valid=false；已记录错误类型为 {json.dumps(dict(error_summary), ensure_ascii=False, sort_keys=True)}。",
        "具体表现包括 `moto/stepfunctions/urls.py` / `moto/stepfunctions/__init__.py` 的 IndentationError，以及 DynamoDB models 导入文本被写成带字面 `\\n` 的 SyntaxError。它们使两个原本 PASS_TO_PASS 测试也失败，属于编辑语义/执行内容破坏共享导入前置条件，不是 callback 修复后的回归。",
        "",
        "逐 session 的动作、工具错误、成功修改、失败 test IDs 与 audit 计数见 `audit.json`。",
    ]
    return "\n".join(lines) + "\n"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise AuditError(f"invalid or empty JSONL: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
