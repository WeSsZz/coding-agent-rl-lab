from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize moto-7607 P0/P1/P2 diagnostic evidence")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--training-dir", required=True)
    args = parser.parse_args()
    evidence = Path(args.evidence_dir).resolve()
    training = Path(args.training_dir).resolve()
    json_output = evidence / "summary.json"
    markdown_output = evidence / "summary.md"
    if json_output.exists() or markdown_output.exists():
        raise SystemExit("refusing to overwrite diagnostic summary")

    p0 = _json(evidence / "protocol-audit-v3.json")
    scores = {
        "A_7b_sft60": _score(evidence / "p1a-A-sft60-target-scores.json"),
        "B_7b_semantic_old": _score(evidence / "p1a-B-semantic-target-scores.json"),
        "C_14b_parent_path": _score(evidence / "p1a-C-14b-parent-target-scores.json"),
    }
    p1_dirs = {
        "A_7b_sft60": evidence / "p1b-A-sft60-greedy-seed123100",
        "B_7b_semantic_old": evidence / "p1b-B-semantic-greedy-seed123100",
        "C_14b_parent_path": evidence / "p1b-C-14b-parent-greedy-seed123100",
    }
    p1 = {name: _evaluation(path) for name, path in p1_dirs.items()}
    curve = {
        "step_4": _score(evidence / "p2-checkpoint-4-target-scores.json"),
        "step_8": _score(evidence / "p2-checkpoint-8-target-scores.json"),
        "step_12": _score(evidence / "p2-checkpoint-12-target-scores.json"),
    }
    p2_eval_dir = evidence / "p2-final-training3-greedy-seed123300"
    p2 = _evaluation(p2_eval_dir)
    training_report = _json(training / "training-report.json")

    adapter_paths = {
        "A_7b_sft60": Path("/root/autodl-tmp/sft-grpo-dynamic-v2-60step-seed73001/final-adapter"),
        "B_7b_semantic_old": Path("/root/autodl-tmp/sft60-semantic-recovery-seed122001/final-adapter"),
        "C_14b_parent_path": Path("/root/autodl-tmp/sft-parent-path-14b-2step-seed115001/final-adapter"),
        "P2_step_4": training / "checkpoint-4",
        "P2_step_8": training / "checkpoint-8",
        "P2_step_12": training / "checkpoint-12",
        "P2_final": training / "final-adapter",
    }
    adapters = {
        name: {
            "path": str(path),
            "weights_sha256": _sha256(path / "adapter_model.safetensors"),
            "config_sha256": _sha256(path / "adapter_config.json"),
        }
        for name, path in adapter_paths.items()
    }

    summary = {
        "schema_version": 1,
        "completed": True,
        "task_id": "getmoto__moto-7607",
        "gpu_budget": "single 48G GPU",
        "p0": {
            "gate": p0["gate"],
            "label_audit": p0["label_audit"],
            "rendering_contract": {
                key: p0["rendering_contract"][key]
                for key in (
                    "sft_rows_have_tools_field",
                    "sft_prompt_tokens",
                    "rollout_prompt_tokens_with_tools",
                    "rendered_prompts_equal",
                    "history_assistant_bare_content_count",
                    "history_assistant_structured_tool_call_count",
                )
            },
            "transition_coverage": p0["transition_coverage"],
        },
        "p1_target_scores": scores,
        "p1_greedy_fixed_states": p1,
        "p2_training": {
            "path": str(training),
            "dataset_sha256": training_report["dataset_sha256"],
            "optimizer_steps": training_report["optimizer_steps"],
            "elapsed_seconds": training_report["elapsed_seconds"],
            "peak_cuda_memory_bytes": training_report["peak_cuda_memory_bytes"],
            "train_loss": training_report["training_metrics"]["train_loss"],
            "final_grad_norm": training_report["training_metrics"]["grad_norm"],
            "adapter_weights_changed": training_report["adapter_weights_changed"],
            "target_score_curve": curve,
        },
        "p2_training_state_gate": p2,
        "adapters": adapters,
        "decision": {
            "selected_branch": "P2 single 7B learnability diagnostic after P1 branch III",
            "p2_required": "3/3 strict success and no new failures",
            "p2_passed": False,
            "reason": "0/3 strict success and 0/3 failure reduction; one trial introduced new failures",
            "p3_started": False,
            "additional_sft_started": False,
            "grpo_started": False,
            "stop_is_pre_registered": True,
        },
        "dependency_note": {
            "observed_system_peft": "0.17.1 incompatible with Transformers 5.16.1",
            "training_overlay": "/root/autodl-tmp/peft-0.21.0-overlay",
            "training_overlay_version": "0.21.0",
            "existing_environment_overwritten": False,
        },
    }
    json_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# moto-7607 P0/P1/P2 诊断执行汇总（2026-09-17）",
        "",
        "## 结论",
        "",
        "本轮按预注册门槛停止在 P2。修复训练/rollout 协议差异后，目标动作 NLL 明显下降，"
        "但最终 adapter 在三个训练状态仍为 0/3 strict、0/3 failure reduction，因此不进入 P3，"
        "不追加 SFT，不启动 GRPO。",
        "",
        "## P0",
        "",
        f"- 修复后 P0 gate：`{p0['gate']['p0_passed']}`。",
        f"- 监督 tokens：{p0['label_audit']['supervised_token_total']}；全 mask、边界错位、截断均为 0。",
        "- SFT 与 rollout 的工具 schema 和历史结构化 tool_calls 渲染哈希一致。",
        "- 完整教师链 18 个动作；旧混合数据只选了 12 个，漏掉的 6 个主要是 search/read。",
        "",
        "## P1 冻结对照",
        "",
    ]
    for name in scores:
        lines.append(
            f"- {name}: target NLL={scores[name]['mean_nll']:.6f}, "
            f"token accuracy={scores[name]['mean_token_accuracy']:.6f}, "
            f"fixed-state strict={p1[name]['summary']['strict_successes']}/"
            f"{p1[name]['summary']['trial_count']}。"
        )
    lines.extend(
        [
            "",
            "## P2 唯一诊断 SFT",
            "",
            f"- 12 steps，耗时 {training_report['elapsed_seconds']:.3f}s，峰值 "
            f"{training_report['peak_cuda_memory_bytes']} bytes，train loss "
            f"{training_report['training_metrics']['train_loss']:.6f}。",
            f"- NLL 曲线：0.572087（稳定 A）→ {curve['step_4']['mean_nll']:.6f}"
            f" → {curve['step_8']['mean_nll']:.6f} → {curve['step_12']['mean_nll']:.6f}。",
            f"- 最终训练状态：strict {p2['summary']['strict_successes']}/"
            f"{p2['summary']['trial_count']}，failure reduction "
            f"{p2['summary']['trials_resolving_failures']}/{p2['summary']['trial_count']}，"
            f"new failures {p2['summary']['trials_with_new_failures']}。",
            "",
            "## 停止决定",
            "",
            "NLL 改善只证明监督信号被学到一部分，没有转化为完整行为恢复。继续增加 steps、换 seed 或"
            "直接进入 GRPO 都没有当前证据支持。下一轮应重新设计多题语义编辑数据和动作级评估，"
            "而不是在这三个状态上继续拟合。",
            "",
            "所有 adapter 路径与 SHA-256、逐状态轨迹、verifier 结果和资源数据见 `summary.json`。",
        ]
    )
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["decision"], ensure_ascii=False, indent=2))


def _score(path: Path) -> dict[str, Any]:
    report = _json(path)
    return {
        "adapter_weights_sha256": report["adapter_weights_sha256"],
        "example_count": report["example_count"],
        "mean_nll": report["mean_nll"],
        "mean_token_accuracy": report["mean_token_accuracy"],
        "by_action_group": report["by_action_group"],
        "peak_cuda_memory_bytes": report["peak_cuda_memory_bytes"],
        "elapsed_seconds": report["elapsed_seconds"],
    }


def _evaluation(path: Path) -> dict[str, Any]:
    report = _json(path / "report.json")
    traces = _jsonl(path / "tool-traces.jsonl")
    actions: dict[str, list[dict[str, Any]]] = {}
    for state in report["states"]:
        state_id = state["state_id"]
        actions[state_id] = [
            row["request"]["action"]
            for row in traces
            if row.get("state_id") == state_id
            and row.get("path", "").endswith("/actions")
            and isinstance(row.get("request", {}).get("action"), dict)
        ]
    return {
        "summary": report["summary"],
        "adapter_unchanged": report["adapter_unchanged"],
        "optimizer_steps": report["optimizer_steps"],
        "peak_cuda_memory_bytes": report.get("peak_cuda_memory_bytes"),
        "states": [
            {
                "state_id": state["state_id"],
                "used_for_training": state["used_for_training"],
                "reward": state["reward"],
                "reward_components": state["reward_components"],
                "actions": actions[state["state_id"]],
            }
            for state in report["states"]
        ],
    }


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if __name__ == "__main__":
    main()
