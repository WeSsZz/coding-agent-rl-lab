from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from coding_agent_rl_lab.grpo_remote import RemoteGRPOCodingEnvironment
from coding_agent_rl_lab.sft_train import prepare_prompt_completion_rows


class ProtocolAuditError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit semantic-recovery SFT labels against the actual rollout rendering"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-report", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--verified-trajectories", required=True)
    parser.add_argument("--verifier-selfcheck", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise ProtocolAuditError(f"refusing to overwrite audit output: {output}")

    try:
        import transformers
        import trl
        from transformers import AutoTokenizer
        from trl.data_utils import _tokenize, apply_chat_template
    except ImportError as exc:
        raise ProtocolAuditError(f"audit dependencies unavailable: {exc}") from exc

    dataset_path = Path(args.dataset)
    report_path = Path(args.dataset_report)
    contexts_path = Path(args.contexts)
    trajectories_path = Path(args.verified_trajectories)
    selfcheck_path = Path(args.verifier_selfcheck)
    rows = _jsonl(dataset_path)
    training_rows = prepare_prompt_completion_rows(rows)
    contexts = _jsonl(contexts_path)
    trajectories = _jsonl(trajectories_path)
    dataset_report = _json_object(report_path)
    verifier_selfcheck = _json_object(selfcheck_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    environment = RemoteGRPOCodingEnvironment(
        "http://127.0.0.1:1", "protocol-audit-token", navigation_first=True
    )
    tools = [
        member
        for name, member in inspect.getmembers(environment, predicate=inspect.ismethod)
        if name not in {"reset", "get_reward"} and not name.startswith("_")
    ]
    tool_names = [tool.__name__ for tool in tools]

    row_audits: list[dict[str, Any]] = []
    boundary_mismatch_count = 0
    target_decode_mismatch_count = 0
    all_mask_count = 0
    over_limit_count = 0
    history_assistant_content_count = 0
    history_structured_tool_call_count = 0
    action_counts: Counter[str] = Counter()
    selected_by_state: dict[str, list[int]] = defaultdict(list)

    for index, (row, training_row) in enumerate(zip(rows, training_rows, strict=True)):
        prompt = training_row["prompt"]
        completion = training_row["completion"]
        row_tools = training_row.get("tools")
        prompt_ids = _tokenize(
            tokenizer,
            prompt,
            add_generation_prompt=True,
            tools=row_tools,
        )["input_ids"]
        full_ids = _tokenize(
            tokenizer,
            [*prompt, *completion],
            tools=row_tools,
        )["input_ids"]
        boundary_matches = full_ids[: len(prompt_ids)] == prompt_ids
        boundary_mismatch_count += int(not boundary_matches)
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        supervised_ids = [token for token in labels if token != -100]
        all_mask_count += int(not supervised_ids)
        over_limit_count += int(len(full_ids) > 16384)
        supervised_text = tokenizer.decode(supervised_ids, skip_special_tokens=False)
        target_content = completion[0]["content"]
        target_present = target_content in supervised_text
        target_decode_mismatch_count += int(not target_present)
        target = row["target_action"]
        action_counts[str(target["kind"])] += 1
        recovery = row.get("semantic_recovery", {})
        state_id = recovery.get("state_id")
        target_index = recovery.get("sequence")
        if isinstance(state_id, str) and isinstance(target_index, int):
            selected_by_state[state_id].append(target_index)

        for message in prompt:
            if message.get("role") == "assistant":
                if isinstance(message.get("tool_calls"), list):
                    history_structured_tool_call_count += 1
                elif isinstance(message.get("content"), str):
                    history_assistant_content_count += 1

        row_audits.append(
            {
                "example_id": row["example_id"],
                "stage": row.get("stage"),
                "state_id": state_id,
                "target_action_index": target_index,
                "target_kind": target["kind"],
                "prompt_tokens": len(prompt_ids),
                "full_tokens": len(full_ids),
                "supervised_tokens": len(supervised_ids),
                "prompt_is_full_prefix": boundary_matches,
                "target_content_in_supervised_decode": target_present,
                "supervised_sha256": _sha256_text(supervised_text),
                "supervised_preview": supervised_text[:500],
            }
        )

    first_prompt = training_rows[0]["prompt"]
    first_tools = training_rows[0].get("tools")
    sft_rendered = tokenizer.apply_chat_template(
        first_prompt,
        tokenize=False,
        add_generation_prompt=True,
        tools=first_tools,
    )
    rollout_rendered = apply_chat_template(
        {"prompt": first_prompt},
        tokenizer,
        tools=tools,
    )["prompt"]
    if not isinstance(sft_rendered, str) or not isinstance(rollout_rendered, str):
        raise ProtocolAuditError("chat template did not return rendered strings")

    context_initial_pairs = {
        (
            context["prompt"][0]["content"],
            context["prompt"][1]["content"],
        )
        for context in contexts
    }
    recovery_initial_pairs = {
        (row["messages"][0]["content"], row["messages"][1]["content"])
        for row in rows
        if row.get("stage") == "semantic-recovery"
    }

    full_teacher_by_state: dict[str, list[str]] = {}
    full_teacher_count = 0
    for trajectory in trajectories:
        kinds = [str(step["action"]["kind"]) for step in trajectory["recovery_steps"]]
        full_teacher_by_state[str(trajectory["state_id"])] = kinds
        full_teacher_count += len(kinds)

    coverage = {}
    for state_id, kinds in full_teacher_by_state.items():
        selected = sorted(selected_by_state.get(state_id, []))
        coverage[state_id] = {
            "teacher_action_count": len(kinds),
            "teacher_action_kinds": kinds,
            "selected_target_indexes": selected,
            "unselected_target_indexes": [i for i in range(len(kinds)) if i not in selected],
            "selected_target_kinds": [kinds[i] for i in selected],
            "unselected_target_kinds": [kinds[i] for i in range(len(kinds)) if i not in selected],
        }

    baseline_status = _parse_test_status(str(verifier_selfcheck["baseline_observation"]))
    final_observation = str(verifier_selfcheck["actions"][-1]["observation"])
    final_status = _parse_test_status(final_observation)

    report = {
        "schema_version": 1,
        "completed": True,
        "versions": {"transformers": transformers.__version__, "trl": trl.__version__},
        "inputs": {
            "model_path": str(Path(args.model_path).resolve()),
            "dataset_sha256": _sha256_file(dataset_path),
            "dataset_report_sha256": _sha256_file(report_path),
            "contexts_sha256": _sha256_file(contexts_path),
            "verified_trajectories_sha256": _sha256_file(trajectories_path),
            "verifier_selfcheck_sha256": _sha256_file(selfcheck_path),
        },
        "dataset": {
            "example_count": len(rows),
            "declared_example_count": dataset_report.get("example_count"),
            "action_counts": dict(sorted(action_counts.items())),
            "row_audits": row_audits,
        },
        "label_audit": {
            "completion_only_loss_reconstructed_from_installed_trl": True,
            "boundary_mismatch_count": boundary_mismatch_count,
            "target_decode_mismatch_count": target_decode_mismatch_count,
            "all_mask_count": all_mask_count,
            "over_16384_count": over_limit_count,
            "supervised_token_total": sum(item["supervised_tokens"] for item in row_audits),
            "supervised_token_min": min(item["supervised_tokens"] for item in row_audits),
            "supervised_token_max": max(item["supervised_tokens"] for item in row_audits),
        },
        "rendering_contract": {
            "rollout_tool_names": tool_names,
            "sft_rows_have_tools_field": all("tools" in row for row in training_rows),
            "sft_prompt_sha256": _sha256_text(sft_rendered),
            "rollout_prompt_sha256_with_tools": _sha256_text(rollout_rendered),
            "sft_prompt_tokens": len(
                tokenizer.encode(sft_rendered, add_special_tokens=False)
            ),
            "rollout_prompt_tokens_with_tools": len(
                tokenizer.encode(rollout_rendered, add_special_tokens=False)
            ),
            "rendered_prompts_equal": sft_rendered == rollout_rendered,
            "sft_prompt_preview": sft_rendered[:1200],
            "rollout_prompt_preview": rollout_rendered[:1200],
            "history_assistant_bare_content_count": history_assistant_content_count,
            "history_assistant_structured_tool_call_count": history_structured_tool_call_count,
        },
        "initial_context": {
            "unique_fixed_context_initial_pairs": len(context_initial_pairs),
            "unique_recovery_initial_pairs": len(recovery_initial_pairs),
            "all_fixed_initial_pairs_present_in_recovery": context_initial_pairs.issubset(
                recovery_initial_pairs
            ),
            "fixed_initial_pair_hashes": sorted(
                _sha256_text(system + "\n" + user) for system, user in context_initial_pairs
            ),
            "recovery_initial_pair_hashes": sorted(
                _sha256_text(system + "\n" + user) for system, user in recovery_initial_pairs
            ),
        },
        "transition_coverage": {
            "teacher_action_total": full_teacher_count,
            "selected_recovery_target_total": sum(len(v) for v in selected_by_state.values()),
            "by_state": coverage,
        },
        "verifier_evidence": {
            "base_commit_expected": verifier_selfcheck.get("base_commit_expected"),
            "base_commit_observed": verifier_selfcheck.get("base_commit_observed"),
            "docker_image_id": verifier_selfcheck.get("docker_image_id"),
            "baseline": baseline_status,
            "final": final_status,
            "strict_success": verifier_selfcheck.get("reward_components", {}).get(
                "strict_success"
            ),
        },
        "gate": {
            "labels_correct": not any(
                [
                    boundary_mismatch_count,
                    target_decode_mismatch_count,
                    all_mask_count,
                    over_limit_count,
                ]
            ),
            "training_rollout_rendering_equal": sft_rendered == rollout_rendered,
            "history_representation_matches_rollout_structured_calls": (
                history_assistant_content_count == 0
                and history_structured_tool_call_count > 0
            ),
            "verifier_positive_and_negative_controls_pass": (
                baseline_status["result"] == "failed"
                and baseline_status["failed_count"] == 1
                and baseline_status["passed_count"] == 2
                and final_status["result"] == "passed"
                and final_status["failed_count"] == 0
                and final_status["passed_count"] == 3
            ),
        },
    }
    report["gate"]["p0_passed"] = all(report["gate"].values())

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "label_audit": report["label_audit"],
                "rendering_contract": {
                    key: report["rendering_contract"][key]
                    for key in (
                        "sft_rows_have_tools_field",
                        "sft_prompt_tokens",
                        "rollout_prompt_tokens_with_tools",
                        "rendered_prompts_equal",
                        "history_assistant_bare_content_count",
                        "history_assistant_structured_tool_call_count",
                    )
                },
                "transition_coverage": report["transition_coverage"],
                "gate": report["gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _parse_test_status(observation: str) -> dict[str, Any]:
    matches = re.findall(
        r"(?m)^(FAILED|PASSED|ERROR)\s+(tests/[^\s]+)", observation
    )
    statuses = {node_id: status for status, node_id in matches}
    return {
        "result": "passed" if observation.startswith("Tests passed") else "failed",
        "failed_count": sum(status == "FAILED" for status in statuses.values()),
        "passed_count": sum(status == "PASSED" for status in statuses.values()),
        "error_count": sum(status == "ERROR" for status in statuses.values()),
        "statuses": dict(sorted(statuses.items())),
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolAuditError(f"expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
