"""Evaluate frozen adapters from replayed real repository states."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from .contracts import AgentAction
from .grpo_evaluate import summarize
from .grpo_remote import RemoteGRPOCodingEnvironment, add_parent_path_evidence
from .grpo_train import configure_tool_response_parsing, probe_bare_json_tool_parsing, read_worker_token
from .lora_inference import merge_lora_adapter_for_inference


class FixedStateEvaluationError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--worker-token-file", required=True)
    parser.add_argument("--worker-base-url", default="http://127.0.0.1:9011")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-completion-length", type=int, default=4096)
    parser.add_argument("--max-tool-calling-iterations", type=int, default=12)
    parser.add_argument("--training-states-only", action="store_true")
    parser.add_argument("--expected-context-count", type=int)
    parser.add_argument("--diagnostic-scope")
    parser.add_argument("--stop-gate")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and not args.resume:
        raise FixedStateEvaluationError(f"refusing to overwrite output directory: {output}")
    if args.resume and not (output / "report.json").is_file():
        raise FixedStateEvaluationError("resume requires an existing report.json")
    contexts_path = Path(args.contexts)
    contexts = _load_contexts(contexts_path)
    if args.training_states_only:
        contexts = [context for context in contexts if context["used_for_training"]]
    expected_context_count = args.expected_context_count
    if expected_context_count is None:
        expected_context_count = 3 if args.training_states_only else 4
    if expected_context_count <= 0:
        raise FixedStateEvaluationError("expected context count must be positive")
    if len(contexts) != expected_context_count:
        raise FixedStateEvaluationError(
            f"fixed-state diagnostic requires exactly {expected_context_count} contexts"
        )
    adapter = Path(args.adapter_path).resolve()
    before = _sha256(adapter / "adapter_model.safetensors")
    token = read_worker_token(Path(args.worker_token_file))
    output.mkdir(parents=True, exist_ok=args.resume)
    manifest = {
        "schema_version": 1,
        "evaluation_kind": "fixed-real-state-continuation-diagnostic",
        "diagnostic_scope": args.diagnostic_scope or (
            "three training-state fit checks only"
            if args.training_states_only
            else (
                "three training-state fit checks plus one unseen-session navigation state; "
                "not held-out edit-only performance"
            )
        ),
        "model_path": str(Path(args.model_path).resolve()),
        "adapter_path": str(adapter),
        "adapter_sha256": before,
        "contexts_path": str(contexts_path.resolve()),
        "contexts_sha256": _sha256(contexts_path),
        "state_ids": [row["state_id"] for row in contexts],
        "seed": args.seed,
        "planned_trial_count": len(contexts),
        "budget": {
            "max_completion_length": args.max_completion_length,
            "max_tool_calling_iterations": args.max_tool_calling_iterations,
            "decoding": "greedy",
            "continuations_per_state": 1,
        },
        "protocol": (
            "TRL bare-JSON tools with navigation-first-v4 parent-path evidence; "
            "tool schemas and structured tool-call history v2"
        ),
        "stop_gate": args.stop_gate or "candidate >=2/4 strict and strictly above SFT60 baseline",
    }
    if args.resume:
        previous_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if previous_manifest != manifest:
            raise FixedStateEvaluationError("resume manifest does not match frozen evaluation")
    else:
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    import torch
    import transformers
    import trl
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    configure_tool_response_parsing(tokenizer, bare_json_tool_calls=True)
    if not probe_bare_json_tool_parsing(
        tokenizer, lambda active, ids, *, prefix: active.parse_response(ids, prefix=prefix)
    ):
        raise FixedStateEvaluationError("bare JSON parser probe failed")
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, local_files_only=True
    )
    model.to("cuda:0")
    adapter_merge = merge_lora_adapter_for_inference(model, adapter)
    model.requires_grad_(False)
    model.eval()

    contexts_by_id = {row["state_id"]: row for row in contexts}
    environments: list[ReplayedStateEnvironment] = []
    active_audit_path = output / "unused.jsonl"
    trace_path = output / "tool-traces.jsonl"

    class TracedReplayedEnvironment(ReplayedStateEnvironment):
        def _request(self, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            response = super()._request(method, path, payload)
            record = {
                "state_id": self.state_id,
                "task_id": payload.get("task_id", self._task_id),
                "method": method,
                "path": path,
                "request": payload,
                "response": response,
            }
            with trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            return response

    def factory() -> ReplayedStateEnvironment:
        environment = TracedReplayedEnvironment(
            args.worker_base_url,
            token,
            contexts_by_id=contexts_by_id,
            reward_audit_path=active_audit_path,
        )
        environments.append(environment)
        return environment

    config = GRPOConfig(
        output_dir=str(output / "trainer"),
        max_steps=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        generation_batch_size=2,
        num_generations=2,
        num_generations_eval=1,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=args.max_tool_calling_iterations,
        temperature=1.0,
        top_p=0.95,
        bf16=True,
        use_vllm=False,
        report_to="none",
        save_strategy="no",
        seed=args.seed,
        log_completions=True,
        num_completions_to_print=1,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=None,
        args=config,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list([contexts[0]]),
        environment_factory=factory,
    )
    trainer.generation_config.do_sample = False
    trainer.generation_config.temperature = None
    trainer.generation_config.top_p = None
    trainer.generation_config.top_k = None
    trainer.generation_config.min_p = None
    trainer.generation_kwargs.update(
        {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "min_p": None,
        }
    )
    if args.resume:
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        for key, value in manifest.items():
            if report.get(key) != value:
                raise FixedStateEvaluationError(f"resume report disagrees on {key}")
    else:
        report = {
            **manifest,
            "training_performed": False,
            "run_complete": False,
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "trl": trl.__version__,
            },
            "adapter_merge": adapter_merge,
            "states": [],
        }
    _checkpoint(output, report)
    completed_state_ids = {entry["state_id"] for entry in report["states"]}
    try:
        for index, context in enumerate(contexts):
            state_id = context["state_id"]
            audit_path = output / f"{state_id}-reward-audit.jsonl"
            if state_id in completed_state_ids:
                records = _jsonl(audit_path)
                if len(records) != 1 or records[0].get("task_id") != context["task_id"]:
                    raise FixedStateEvaluationError(f"{state_id}: completed audit is invalid")
                continue
            for environment in environments:
                environment._reward_audit_path = audit_path
            active_audit_path = audit_path
            state_seed = args.seed + index * 100
            set_seed(state_seed)
            started = time.monotonic()
            with (output / "generation.log").open("a", encoding="utf-8") as generation_log:
                generation_log.write(f"\n===== state={state_id} seed={state_seed} =====\n")
                with contextlib.redirect_stdout(generation_log):
                    metrics = trainer.evaluate(eval_dataset=Dataset.from_list([context]))
            records = _jsonl(audit_path)
            if len(records) != 1:
                raise FixedStateEvaluationError(
                    f"{state_id}: expected one reward audit, found {len(records)}"
                )
            entry = {
                "state_id": state_id,
                "source_session_id": context["source_session_id"],
                "used_for_training": context["used_for_training"],
                "prefix_action_count": context["prefix_action_count"],
                "seed": state_seed,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "summary": summarize(records),
                "reward": records[0]["reward"],
                "reward_components": records[0]["reward_components"],
                "metrics": metrics,
            }
            report["states"].append(entry)
            report["completed_trial_count"] = len(report["states"])
            _checkpoint(output, report)
        all_records = [
            _jsonl(output / f"{context['state_id']}-reward-audit.jsonl")[0]
            for context in contexts
        ]
        report["summary"] = summarize(all_records)
        report["adapter_unchanged"] = before == _sha256(adapter / "adapter_model.safetensors")
        report["optimizer_steps"] = trainer.state.global_step
        report["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        if not report["adapter_unchanged"] or report["optimizer_steps"] != 0:
            raise FixedStateEvaluationError("evaluation mutated weights or performed optimizer steps")
        report["run_complete"] = True
        _checkpoint(output, report)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    finally:
        for environment in environments:
            environment._delete()


class ReplayedStateEnvironment(RemoteGRPOCodingEnvironment):
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        contexts_by_id: dict[str, dict[str, Any]],
        reward_audit_path: Path,
    ) -> None:
        super().__init__(
            base_url,
            token,
            reward_audit_path=reward_audit_path,
            navigation_first=True,
        )
        self.contexts_by_id = contexts_by_id
        self.state_id: str | None = None

    def reset(self, **kwargs: Any) -> str:
        state_id = kwargs.get("state_id")
        task_id = kwargs.get("task_id")
        if not isinstance(state_id, str) or state_id not in self.contexts_by_id:
            raise FixedStateEvaluationError("reset requires a known state_id")
        self.state_id = state_id
        context = self.contexts_by_id[state_id]
        super().reset(task_id=task_id)
        expected_tools = context["prompt"][3::2]
        for sequence, (raw_action, expected_tool) in enumerate(
            zip(context["prefix_actions"], expected_tools, strict=True)
        ):
            action = AgentAction.from_dict(raw_action)
            observation = getattr(self, action.kind.value)(**action.arguments)
            expected = expected_tool["content"]
            if action.kind.value == "run_tests":
                matches = _test_status(expected) == _test_status(observation)
            else:
                matches = expected.rstrip() == observation.rstrip()
            if not matches:
                raise FixedStateEvaluationError(
                    f"{state_id}: prefix observation changed at action {sequence}"
                )
            if self._completed:
                raise FixedStateEvaluationError(f"{state_id}: prefix terminated during reset")
        return ""


def _load_contexts(path: Path) -> list[dict[str, Any]]:
    contexts = _jsonl(path)
    state_ids: set[str] = set()
    for index, row in enumerate(contexts, start=1):
        if row.get("schema_version") != 1 or row.get("task_id") != "getmoto__moto-7607":
            raise FixedStateEvaluationError(f"invalid context row {index}")
        state_id = row.get("state_id")
        if not isinstance(state_id, str) or not state_id or state_id in state_ids:
            raise FixedStateEvaluationError(f"duplicate or invalid state on row {index}")
        state_ids.add(state_id)
        if not isinstance(row.get("prefix_actions"), list) or len(row["prefix_actions"]) != row.get(
            "prefix_action_count"
        ):
            raise FixedStateEvaluationError(f"invalid prefix actions on row {index}")
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or len(prompt) != 2 + 2 * row["prefix_action_count"]:
            raise FixedStateEvaluationError(f"invalid prompt history on row {index}")
    return contexts


def _test_status(observation: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    result = "passed" if observation.startswith("Tests passed") else "failed"
    statuses = tuple(
        sorted(
            (status, node_id)
            for status, node_id in re.findall(
                r"(?m)^(FAILED|PASSED|ERROR)\s+(tests/[^\s]+)", observation
            )
        )
    )
    return result, statuses


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _checkpoint(output: Path, report: dict[str, Any]) -> None:
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")


if __name__ == "__main__":
    main()
