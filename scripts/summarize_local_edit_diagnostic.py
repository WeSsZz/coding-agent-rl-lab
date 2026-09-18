from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize frozen local-edit diagnostic evidence")
    parser.add_argument("--d1-report", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()

    report = json.loads(Path(args.d1_report).read_text(encoding="utf-8"))
    if not report.get("run_complete") or report.get("summary", {}).get("trial_count") != 7:
        raise RuntimeError("D1 report is incomplete")
    states = [_state_summary(row) for row in report["states"]]
    summary = {
        "schema_version": 1,
        "decision": "stop-before-d2-training-or-grpo",
        "reason": "D1 produced zero verifier-improving edits",
        "adapter_sha256": report["adapter_sha256"],
        "adapter_unchanged": report["adapter_unchanged"],
        "seed": report["seed"],
        "contexts_sha256": report["contexts_sha256"],
        "peak_cuda_memory_bytes": report["peak_cuda_memory_bytes"],
        "counts": report["summary"],
        "failure_categories": {
            category: sum(row["failure_category"] == category for row in states)
            for category in sorted({row["failure_category"] for row in states})
        },
        "states": states,
        "d2_executed": False,
        "training_executed": False,
        "grpo_executed": False,
    }
    output_json = Path(args.output_json)
    output_markdown = Path(args.output_markdown)
    for target in (output_json, output_markdown):
        if target.exists():
            raise RuntimeError(f"refusing to overwrite output: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_markdown.write_text(_markdown(summary) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _state_summary(row: dict[str, Any]) -> dict[str, Any]:
    generated = row.get("generated_action")
    target = row["teacher_target_action"]
    if generated is None:
        category = "unparseable-tool-action"
    elif generated["kind"] not in {"replace_text", "replace_lines"}:
        category = "wrong-tool"
    elif not row["edit_applied"]:
        category = "edit-not-applied"
    elif row["verifier_improving_edit"]:
        category = "verifier-improving"
    else:
        category = "applied-but-semantically-wrong"
    components = row.get("reward_components") or {}
    return {
        "state_id": row["state_id"],
        "source_state_id": row["source_state_id"],
        "teacher_recovery_sequence": row["teacher_recovery_sequence"],
        "failure_category": category,
        "completion_token_count": row["completion_token_count"],
        "hit_max_new_tokens": row["hit_max_new_tokens"],
        "parse_error": row.get("parse_error"),
        "generated_action": generated,
        "teacher_target_action": target,
        "teacher_tool_kind_match": row["teacher_tool_kind_match"],
        "edit_applied": row["edit_applied"],
        "target_file_edit_applied": row["target_file_edit_applied"],
        "observation": row.get("observation"),
        "resolved_failure_count": components.get("resolved_failure_count"),
        "new_failure_count": components.get("new_failure_count"),
        "strict_success": row["strict_success"],
        "completion_text": row["completion_text"],
    }


def _markdown(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        "# moto-7607 局部编辑诊断 D1 结果",
        "",
        "固定使用 P2 step12 adapter，单次 greedy、7 个真实编辑前状态；未训练。",
        "",
        "## 结论",
        "",
        "D1 没有产生任何 verifier-improving 编辑，因此按预注册门槛停止；D2、SFT 和 GRPO 均未启动。",
        "",
        "## 汇总",
        "",
        f"- 可解析动作：{counts['parsed_action_count']}/7",
        f"- 实际编辑动作：{counts['edit_action_count']}/7",
        f"- 成功应用补丁：{counts['edit_applied_count']}/7",
        f"- 应用于教师目标文件：{counts['target_file_edit_applied_count']}/7",
        f"- verifier failure reduction：{counts['verifier_improving_edit_count']}/7",
        f"- strict success：{counts['strict_success_count']}/7",
        f"- 精确教师动作一致：{counts['exact_teacher_action_match_count']}/7（仅辅助指标）",
        f"- 峰值显存：{summary['peak_cuda_memory_bytes']} bytes",
        f"- adapter 未变化：{summary['adapter_unchanged']}",
        "",
        "## 每状态归因",
        "",
    ]
    for row in summary["states"]:
        generated = row["generated_action"]
        generated_label = "unparsed" if generated is None else generated["kind"]
        generated_path = "" if generated is None else str(generated.get("arguments", {}).get("path", ""))
        lines.append(
            f"- `{row['state_id']}`：{row['failure_category']}；生成 `{generated_label}` "
            f"`{generated_path}`；applied={row['edit_applied']}，resolved={row['resolved_failure_count']}，"
            f"new={row['new_failure_count']}，tokens={row['completion_token_count']}，"
            f"truncated={row['hit_max_new_tokens']}。"
        )
    lines.extend(
        [
            "",
            "## 后续含义",
            "",
            "当前瓶颈已经定位到给足正确执行状态后的下一步动作生成：既有协议解析失败，也有补丁可应用但语义不能修复。"
            "在这个门槛下，扩大完整轨迹或训练搜索策略缺乏依据。下一批工作应先围绕可执行局部编辑建立多题留出诊断，"
            "并补干净 7B 底座同状态对照，再决定是否构建新的 recovery SFT。",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
