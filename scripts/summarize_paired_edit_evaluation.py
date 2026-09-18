from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paired base/adapter edit evaluation")
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()
    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("run_complete") or not report.get("teacher_gate_passed"):
        raise RuntimeError("paired report is incomplete or failed its teacher gate")
    by_condition = {row["condition"]: row for row in report["conditions"]}
    base = by_condition["clean-base"]
    adapted = by_condition["p2-step12-adapter"]
    teacher_by_id = {row["state_id"]: row for row in report["teacher_controls"]}
    adapted_by_id = {row["state_id"]: row for row in adapted["states"]}
    states = []
    for base_row in base["states"]:
        state_id = base_row["state_id"]
        states.append(
            {
                "state_id": state_id,
                "teacher_sequence": base_row["teacher_recovery_sequence"],
                "teacher_immediate_strict": teacher_by_id[state_id]["immediate"]["strict_success"],
                "teacher_immediate_resolved": teacher_by_id[state_id]["immediate"][
                    "resolved_failure_count"
                ],
                "teacher_suffix_strict": teacher_by_id[state_id]["with_teacher_suffix"][
                    "strict_success"
                ],
                "clean_base": _generation_summary(base_row),
                "adapter": _generation_summary(adapted_by_id[state_id]),
            }
        )
    summary = {
        "schema_version": 1,
        "decision": "stop-training-and-build-multi-task-local-edit-diagnostic",
        "interpretation": (
            "teacher positive controls validate suffix scoring; the adapter improves tool-format "
            "generation relative to clean base but neither condition produces an execution-semantic "
            "success"
        ),
        "report_sha256": _sha256(report_path),
        "contexts_sha256": report["contexts_sha256"],
        "trajectories_sha256": report["trajectories_sha256"],
        "adapter_sha256": report["adapter_sha256"],
        "adapter_unchanged": report["adapter_unchanged"],
        "seed": report["seed"],
        "teacher": report["teacher_summary"],
        "clean_base": base["summary"],
        "adapter": adapted["summary"],
        "paired": {
            "base_semantic_wins": report["paired_comparison"]["base_semantic_wins"],
            "adapter_semantic_wins": report["paired_comparison"]["adapter_semantic_wins"],
            "base_fewer_special_token_degenerations": report["paired_comparison"][
                "base_fewer_special_token_degenerations"
            ],
        },
        "runtime": report["runtime"],
        "base_condition_adapter_loaded": base["adapter_loaded"],
        "base_condition_model_has_peft_config": base["model_has_peft_config"],
        "peak_cuda_memory_bytes": report["peak_cuda_memory_bytes"],
        "states": states,
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


def _generation_summary(row: dict[str, Any]) -> dict[str, Any]:
    text = row["completion_text"]
    return {
        "parsed": row["generated_action"] is not None,
        "generated_action": row["generated_action"],
        "edit_applied": row["immediate"]["first_edit_applied"],
        "immediate_strict": row["immediate"]["strict_success"],
        "immediate_resolved": row["immediate"]["resolved_failure_count"],
        "teacher_suffix_strict": row["with_teacher_suffix"]["strict_success"],
        "execution_semantic_success": row["execution_semantic_success"],
        "completion_token_count": row["completion_token_count"],
        "termination_reason": row["termination_reason"],
        "special_token_degeneration": row["special_token_degeneration"],
        "special_tokens": row["special_tokens"],
        "completion_token_ids_sha256": hashlib.sha256(
            json.dumps(row["completion_token_ids"]).encode("utf-8")
        ).hexdigest(),
        "completion_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "completion_preview": text[:500],
        "parse_error": row["parse_error"],
    }


def _markdown(summary: dict[str, Any]) -> str:
    teacher = summary["teacher"]
    base = summary["clean_base"]
    adapted = summary["adapter"]
    lines = [
        "# moto-7607 干净 7B 与 adapter 配对评测",
        "",
        "## 结论",
        "",
        "教师单步立即 verifier 只有 3/7 strict，但教师固定后缀为 7/7 strict；因此中间编辑不能只按即时 failure reduction 判错。",
        "",
        "干净底座没有生成可解析工具动作，特殊 token 退化比 adapter 更严重。adapter 改善了协议格式，但两条件在教师后缀语义判定上均为 0/7。"
        "现有证据不支持‘adapter 导致特殊 token 退化’，也不支持继续单题训练。",
        "",
        "## 汇总",
        "",
        f"- 教师目标编辑可应用：{teacher['target_edit_applied_count']}/7",
        f"- 教师立即 strict：{teacher['immediate_strict_success_count']}/7",
        f"- 教师后缀 strict：{teacher['suffix_strict_success_count']}/7",
        f"- 干净底座可解析/可应用/语义成功：{base['parsed_action_count']}/7、{base['edit_applied_count']}/7、{base['execution_semantic_success_count']}/7",
        f"- adapter 可解析/可应用/语义成功：{adapted['parsed_action_count']}/7、{adapted['edit_applied_count']}/7、{adapted['execution_semantic_success_count']}/7",
        f"- 特殊 token 退化：底座 {base['special_token_degeneration_count']}/7；adapter {adapted['special_token_degeneration_count']}/7",
        f"- 达到 1024 token 上限：底座 {base['max_new_tokens_count']}/7；adapter {adapted['max_new_tokens_count']}/7",
        f"- 峰值显存：{summary['peak_cuda_memory_bytes']} bytes",
        f"- 纯底座确认未加载 adapter：{not summary['base_condition_adapter_loaded']}",
        "",
        "## 逐状态",
        "",
    ]
    for row in summary["states"]:
        b = row["clean_base"]
        a = row["adapter"]
        lines.append(
            f"- `{row['state_id']}`：教师立即 strict={row['teacher_immediate_strict']}，"
            f"后缀 strict={row['teacher_suffix_strict']}；底座 parsed={b['parsed']}、"
            f"special={b['special_token_degeneration']}、stop={b['termination_reason']}、"
            f"semantic={b['execution_semantic_success']}；adapter parsed={a['parsed']}、"
            f"special={a['special_token_degeneration']}、stop={a['termination_reason']}、"
            f"semantic={a['execution_semantic_success']}。"
        )
    lines.extend(
        [
            "",
            "## 决策",
            "",
            "保持训练、完整 rollout 和 GRPO 暂停。下一步应建立按题切分的 20–30 题局部编辑诊断集，并在多题上复用教师后缀判定。"
            "同时单独审计当前 bare-JSON 工具协议与 Qwen 原生 chat template 的兼容性；特殊 token 已存在于模型原始生成 ID 中，不是 parser 或显示层伪造。",
        ]
    )
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if __name__ == "__main__":
    main()
