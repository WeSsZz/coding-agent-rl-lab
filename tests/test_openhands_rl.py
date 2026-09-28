from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from coding_agent_rl_lab.openhands_rl import RolloutError, load_rollout, validate_group


class OpenHandsRLTests(unittest.TestCase):
    def make_rollout(self, root: Path, run_id: str, reward: int):
        trace = root / f"{run_id}.jsonl"
        row = {
            "run_id": run_id, "instance_id": "getmoto__moto-7646",
            "policy_revision": "qwen3-checkpoint-A", "sequence": 1, "status": 200,
            "request": {"temperature": 1.0, "top_p": 1.0,
                        "messages": [{"role": "user", "content": "Fix the task"}]},
            "response": {
                "id": f"completion-{run_id}", "model": "qwen3-base",
                "prompt_token_ids": [1, 2, 3],
                "choices": [{"token_ids": [4, 5], "logprobs": {
                    "content": [{"token": "token_id:4", "logprob": -0.2},
                                {"token": "token_id:5", "logprob": -0.3}],
                }}],
            },
        }
        trace.write_text(json.dumps(row) + "\n", encoding="utf-8")
        manifest = {
            "run_id": run_id, "instance_id": row["instance_id"],
            "policy_revision": row["policy_revision"], "model_id": "qwen3-base",
            "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            "rollout_sha256": "a" * 64, "verifier_sha256": "b" * 64,
            "git_patch_sha256": "c" * 64,
            "generation_count": 1, "response_ids": [f"completion-{run_id}"],
            "tool_boundaries": [{"response_id": f"completion-{run_id}", "tool_calls": []}],
            "sampled_token_count": 2, "reward": reward,
            "verifier_report": {"resolved": bool(reward), "error_eval": False},
            "rollout_error": None,
        }
        joined = root / f"{run_id}.reward.json"
        joined.write_text(json.dumps(manifest), encoding="utf-8")
        return trace, joined

    def test_group_preflight_and_trace_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.make_rollout(root, "run-1", 0)
            second = self.make_rollout(root, "run-2", 1)
            rollouts = [load_rollout(*first), load_rollout(*second)]
            validate_group(rollouts, "qwen3-checkpoint-A")
            self.assertEqual([item.sampled_tokens for item in rollouts], [2, 2])
            with self.assertRaisesRegex(RolloutError, "Duplicate rollout"):
                validate_group([rollouts[0], rollouts[0]], "qwen3-checkpoint-A")
            with self.assertRaisesRegex(RolloutError, "constant"):
                validate_group([rollouts[0], rollouts[0].__class__(
                    "run-3", rollouts[0].instance_id, rollouts[0].policy_revision,
                    rollouts[0].model_id, 0, rollouts[0].generations, rollouts[0].reward_path,
                )], "qwen3-checkpoint-A")
            with self.assertRaisesRegex(RolloutError, "revision"):
                validate_group(rollouts, "wrong-checkpoint")
            first[0].write_text(first[0].read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RolloutError, "digest"):
                load_rollout(*first)

    def test_missing_prompt_ids_and_verifier_error_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace, joined = self.make_rollout(root, "run-1", 1)
            manifest = json.loads(joined.read_text(encoding="utf-8"))
            manifest["verifier_report"]["error_eval"] = True
            joined.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RolloutError, "infrastructure"):
                load_rollout(trace, joined)
            manifest["verifier_report"]["error_eval"] = False
            row = json.loads(trace.read_text(encoding="utf-8"))
            del row["response"]["prompt_token_ids"]
            trace.write_text(json.dumps(row), encoding="utf-8")
            manifest["trace_sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
            joined.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RolloutError, "Prompt"):
                load_rollout(trace, joined)

    def test_sampling_mismatch_rejected_but_model_loop_is_valid_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace, joined = self.make_rollout(root, "run-loop", 0)
            manifest = json.loads(joined.read_text(encoding="utf-8"))
            manifest["rollout_error"] = "Agent got stuck in a loop"
            joined.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(load_rollout(trace, joined).reward, 0)
            row = json.loads(trace.read_text(encoding="utf-8"))
            row["request"]["temperature"] = 0.7
            trace.write_text(json.dumps(row), encoding="utf-8")
            manifest["trace_sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
            joined.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RolloutError, "temperature-1"):
                load_rollout(trace, joined)

    def test_official_completion_and_verifier_join(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace, _ = self.make_rollout(root, "run-join", 0)
            rollout_path = root / "output.jsonl"
            verifier_path = root / "verifier.jsonl"
            patch = ""
            rollout_path.write_text(json.dumps({
                "instance_id": "getmoto__moto-7646",
                "test_result": {"git_patch": patch}, "history": [],
            }) + "\n", encoding="utf-8")
            verifier_path.write_text(json.dumps({
                "instance_id": "getmoto__moto-7646",
                "test_result": {"git_patch": patch, "report": {
                    "resolved": False, "error_eval": False, "test_timeout": False,
                }},
            }) + "\n", encoding="utf-8")
            completions = root / "llm_completions" / "getmoto__moto-7646"
            completions.mkdir(parents=True)
            (completions / "000.json").write_text(json.dumps({
                "response": {"id": "completion-run-join"},
                "fncall_response": {"choices": [{"message": {"tool_calls": []}}]},
            }), encoding="utf-8")
            joined = root / "joined.json"
            script = Path(__file__).resolve().parents[1] / "scripts/vm/join_swegym_rollout_reward.py"
            result = subprocess.run([
                sys.executable, str(script), "--trace", str(trace),
                "--rollout", str(rollout_path), "--verifier", str(verifier_path),
                "--completions-dir", str(completions), "--output", str(joined),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(load_rollout(trace, joined).reward, 0)


if __name__ == "__main__":
    unittest.main()
