"""One-step, verifier-reward policy-gradient smoke for official OpenHands rollouts.

This intentionally does not use the legacy remote GRPO environment. Rollouts are
collected by the pinned SWE-Gym OpenHands fork before this process is started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


class RolloutError(ValueError):
    pass


@dataclass(frozen=True)
class Generation:
    prompt_ids: list[int]
    sampled_ids: list[int]
    sampled_logprobs: list[float]


@dataclass(frozen=True)
class Rollout:
    run_id: str
    instance_id: str
    policy_revision: str
    model_id: str
    reward: int
    generations: list[Generation]
    reward_path: Path

    @property
    def sampled_tokens(self) -> int:
        return sum(len(item.sampled_ids) for item in self.generations)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rollout(trace_path: Path, reward_path: Path) -> Rollout:
    manifest = json.loads(reward_path.read_text(encoding="utf-8"))
    if manifest.get("trace_sha256") != _digest(trace_path):
        raise RolloutError(f"Trace digest does not match joined reward: {trace_path}")
    if not all(isinstance(manifest.get(key), str) and len(manifest[key]) == 64
               for key in ("rollout_sha256", "verifier_sha256", "git_patch_sha256")):
        raise RolloutError("Joined reward lacks official rollout/verifier provenance")
    rollout_error = manifest.get("rollout_error")
    if rollout_error and not any(marker in str(rollout_error).lower() for marker in (
        "stuck in a loop", "maximum iteration",
    )):
        raise RolloutError("OpenHands rollout reported an unclassified error")
    report = manifest.get("verifier_report", {})
    if report.get("error_eval") or report.get("test_timeout"):
        raise RolloutError("Verifier infrastructure error is not a reward")
    reward = manifest.get("reward")
    if type(reward) is not int or reward not in (0, 1) or reward != int(report.get("resolved") is True):
        raise RolloutError("Reward does not match clean verifier result")
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or len(rows) != manifest.get("generation_count"):
        raise RolloutError("Trace generation count does not match joined reward")
    generations: list[Generation] = []
    response_ids: list[str] = []
    for number, row in enumerate(rows, 1):
        if row.get("sequence") != number or row.get("status") != 200:
            raise RolloutError("Trace has failed or unordered completions")
        request = row.get("request", {})
        if (request.get("temperature") != 1.0 or request.get("top_p", 1.0) != 1.0 or
                request.get("n", 1) != 1 or request.get("stream", False) or
                any(request.get(key, 0) != 0 for key in (
                    "frequency_penalty", "presence_penalty"
                )) or request.get("repetition_penalty", 1) != 1 or
                request.get("min_p", 0) != 0 or request.get("top_k", -1) != -1 or
                any(request.get(key) for key in ("logit_bias", "guided_decoding", "bad_words"))):
            raise RolloutError("Replay requires unmodified temperature-1 sampling")
        if any(row.get(key) != manifest.get(key) for key in (
            "run_id", "instance_id", "policy_revision"
        )):
            raise RolloutError("Trace identity differs from joined reward")
        response = row["response"]
        if response.get("model") != manifest.get("model_id"):
            raise RolloutError("Served model differs from joined reward")
        if len(response.get("choices", [])) != 1:
            raise RolloutError("Expected one sampled choice per model call")
        choice = response["choices"][0]
        prompt = response.get("prompt_token_ids")
        sampled = choice.get("token_ids")
        content = (choice.get("logprobs") or {}).get("content")
        if not isinstance(prompt, list) or not prompt or not isinstance(sampled, list) or not sampled:
            raise RolloutError("Prompt or sampled token IDs are missing")
        if not all(type(token) is int and token >= 0 for token in prompt + sampled):
            raise RolloutError("Invalid token ID")
        if not isinstance(content, list) or len(sampled) != len(content):
            raise RolloutError("Sampled logprobs are missing or misaligned")
        logprobs = [item.get("logprob") for item in content]
        if not all(type(value) in (int, float) and math.isfinite(value) for value in logprobs):
            raise RolloutError("Invalid sampled logprob")
        if any(item.get("token") != f"token_id:{token}"
               for item, token in zip(content, sampled)):
            raise RolloutError("Sampled logprobs do not identify sampled token IDs")
        response_ids.append(response["id"])
        generations.append(Generation(prompt, sampled, logprobs))
    if response_ids != manifest.get("response_ids") or len(set(response_ids)) != len(rows):
        raise RolloutError("Response IDs differ from joined official completions")
    if sum(len(item.sampled_ids) for item in generations) != manifest.get("sampled_token_count"):
        raise RolloutError("Sampled token count differs from joined reward")
    boundaries = manifest.get("tool_boundaries")
    if not isinstance(boundaries, list) or [entry.get("response_id") for entry in boundaries] != response_ids:
        raise RolloutError("Official tool boundaries do not match model responses")
    return Rollout(
        manifest["run_id"], manifest["instance_id"], manifest["policy_revision"],
        manifest["model_id"], reward, generations, reward_path,
    )


def validate_group(rollouts: list[Rollout], policy_revision: str) -> None:
    if len(rollouts) < 2:
        raise RolloutError("At least two independent rollouts are required")
    if len({item.run_id for item in rollouts}) != len(rollouts):
        raise RolloutError("Duplicate rollout ID")
    if len({item.instance_id for item in rollouts}) != 1:
        raise RolloutError("Smoke group must use one fixed SWE-Gym train task")
    if {item.policy_revision for item in rollouts} != {policy_revision}:
        raise RolloutError("Rollouts were not sampled from the declared policy revision")
    if len({item.model_id for item in rollouts}) != 1:
        raise RolloutError("Rollouts used different served model IDs")
    if len({item.reward for item in rollouts}) < 2:
        raise RolloutError("Rewards are constant; a policy-gradient update would be zero")


def train_one_step(rollouts: list[Rollout], model_path: Path, output_dir: Path,
                   learning_rate: float) -> dict:
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RolloutError("CUDA is required for the optimizer smoke")
    if output_dir.exists():
        raise RolloutError(f"Output directory already exists: {output_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear",
    )).cuda()
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=learning_rate)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    parameter_before = {name: p.detach().cpu().clone() for name, p in trainable}
    mean_reward = sum(item.reward for item in rollouts) / len(rollouts)
    losses: list[float] = []
    for rollout in rollouts:
        advantage = rollout.reward - mean_reward
        for generation in rollout.generations:
            # The last prompt token predicts the first sampled token. No
            # prompt or tool-observation token contributes to the loss.
            input_ids = generation.prompt_ids + generation.sampled_ids[:-1]
            if len(input_ids) > model.config.max_position_embeddings:
                raise RolloutError("Replay exceeds model context window")
            if max(input_ids + generation.sampled_ids) >= model.config.vocab_size:
                raise RolloutError("Replay token ID exceeds model vocabulary")
            inputs = torch.tensor([input_ids], dtype=torch.long, device="cuda")
            targets = torch.tensor(generation.sampled_ids, dtype=torch.long, device="cuda")
            outputs = model(input_ids=inputs, logits_to_keep=len(generation.sampled_ids))
            logits = outputs.logits[0]
            if logits.shape[0] != len(generation.sampled_ids):
                raise RolloutError("Model returned misaligned completion logits")
            token_nll = F.cross_entropy(logits.float(), targets, reduction="sum")
            loss = token_nll * (advantage / (len(rollouts) * rollout.sampled_tokens))
            loss.backward()
            losses.append(float(loss.detach().cpu()))
            del outputs, logits, token_nll, inputs, targets, loss
    grad_norm = float(torch.nn.utils.clip_grad_norm_(
        (p for p in model.parameters() if p.requires_grad), 1.0,
    ))
    if not math.isfinite(grad_norm) or grad_norm <= 0:
        raise RolloutError("No finite nonzero policy gradient; refusing optimizer step")
    optimizer.step()
    parameter_delta = max(
        float((p.detach().cpu() - parameter_before[name]).abs().max())
        for name, p in trainable
    )
    if parameter_delta <= 0 or not math.isfinite(parameter_delta):
        raise RolloutError("Optimizer did not change trainable parameters")
    output_dir.mkdir(parents=True)
    model.save_pretrained(output_dir / "adapter")
    return {"optimizer_steps": 1, "grad_norm": grad_norm,
            "parameter_max_delta": parameter_delta,
            "policy_loss": sum(losses), "output_dir": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", action="append", nargs=2, metavar=("TRACE", "REWARD"), required=True)
    parser.add_argument("--policy-revision", required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    rollouts = [load_rollout(Path(trace), Path(reward)) for trace, reward in args.rollout]
    validate_group(rollouts, args.policy_revision)
    if args.learning_rate <= 0:
        parser.error("Learning rate must be positive")
    result = {"instance_id": rollouts[0].instance_id,
              "policy_revision": args.policy_revision,
              "run_ids": [item.run_id for item in rollouts],
              "rewards": [item.reward for item in rollouts],
              "sampled_tokens": [item.sampled_tokens for item in rollouts],
              "optimizer_steps": 0}
    if args.train:
        if args.model_path is None or args.output_dir is None:
            parser.error("--train requires --model-path and --output-dir")
        result.update(train_one_step(rollouts, args.model_path, args.output_dir,
                                     args.learning_rate))
        (args.output_dir / "training-report.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8",
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
