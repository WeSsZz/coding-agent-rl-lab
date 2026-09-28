"""Validated complete-trajectory SFT entry.

Default: CPU data check. ``--train`` runs a one-step GPU smoke and
``--formal-one-epoch`` runs one deterministic pass over every accepted
trajectory.  Both modes preserve complete conversations and their masks.

This entry deliberately does not launch a formal run before the runtime is
validated. It uses the same tokenized conversations intended for later SFT.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .trajectory_sft import pad_training_batch, TrajectoryError


def load_training_data(directory: Path, model: dict):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    raw = (directory / "train.jsonl").read_bytes()
    if manifest.get("schema") != "complete-trajectory-tokens-v1":
        raise TrajectoryError("wrong_training_schema")
    if manifest.get("model") != model["id"] or manifest.get("revision") != model["revision"]:
        raise TrajectoryError("training_model_mismatch")
    if hashlib.sha256(raw).hexdigest() != manifest["data_sha256"]:
        raise TrajectoryError("training_data_hash_mismatch")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not rows or len(rows) != manifest["trajectory_count"]:
        raise TrajectoryError("training_count_mismatch")
    seen, token_total, target_total = set(), 0, 0
    for row in rows:
        if row["instance_id"] in seen:
            raise TrajectoryError("duplicate_training_instance")
        seen.add(row["instance_id"])
        ids, labels, mask = (row[k] for k in ("input_ids", "labels", "attention_mask"))
        if not 1 < len(ids) <= manifest["max_length"] or not len(ids) == len(labels) == len(mask):
            raise TrajectoryError("invalid_training_lengths")
        if any(type(v) is not int or v < 0 for v in ids):
            raise TrajectoryError("invalid_token_id")
        if any(v != 1 for v in mask) or any(type(v) is not int or v not in (-100, ids[i]) for i, v in enumerate(labels)):
            raise TrajectoryError("invalid_training_mask")
        targets = sum(v != -100 for v in labels[1:])
        if not targets or labels[0] != -100:
            raise TrajectoryError("no_causal_targets")
        token_total += len(ids)
        target_total += targets
    if (token_total, target_total) != (manifest["total_tokens"], manifest["total_supervised_tokens"]):
        raise TrajectoryError("training_totals_mismatch")
    return rows, manifest


def select_smoke_row(rows, max_tokens=None):
    """Select the longest complete trajectory within an optional hard cap."""
    if max_tokens is not None:
        if type(max_tokens) is not int or max_tokens < 2:
            raise TrajectoryError("invalid_smoke_token_cap")
        candidates = [row for row in rows if len(row["input_ids"]) <= max_tokens]
    else:
        candidates = rows
    if not candidates:
        raise TrajectoryError("no_complete_trajectory_under_smoke_cap")
    return max(candidates, key=lambda row: len(row["input_ids"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--model-manifest", type=Path, help="pinned model selection; defaults to route v2")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--formal-one-epoch", action="store_true")
    parser.add_argument("--max-smoke-tokens", type=int,
                        help="hard cap for the complete trajectory used by --train; never truncates")
    args = parser.parse_args()
    if args.train and args.formal_one_epoch:
        parser.error("choose either --train or --formal-one-epoch")
    root = Path(__file__).resolve().parents[2]
    model_manifest = args.model_manifest or root / "manifests/route-v2-model.json"
    from scripts.model_contract import selected_model
    model_spec = selected_model(model_manifest)
    rows, manifest = load_training_data(args.data, model_spec)
    for name, key in (("envs/openhands-v0.54-tools.json", "tools_sha256"),
                      ("manifests/route-v2-task-split.json", "split_sha256")):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != manifest[key]:
            raise SystemExit("data contract changed: " + name)
    if args.output.exists():
        raise SystemExit("output already exists")
    summary = {"mode": "data_preflight", "trajectories": len(rows),
               "data_sha256": manifest["data_sha256"], "training_performed": False,
               "runtime_checked": False}
    if not args.train and not args.formal_one_epoch:
        args.output.mkdir(parents=True)
        (args.output / "preflight.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary))
        return
    if args.model_path is None or args.inventory is None:
        parser.error("--train requires --model-path and --inventory")
    if os.environ.get("TRAJECTORY_SFT_BOUNDED") != "1":
        raise SystemExit("use scripts/gpu/run_trajectory_sft_smoke.sh for the billing-time timeout")
    started = int(os.environ["SESSION_STARTED_AT"])
    if not started <= time.time() < started + 14400:
        raise SystemExit("invalid or expired billing window")
    subprocess.run([sys.executable, str(root / "scripts/model_contract.py"),
                    "--manifest", str(model_manifest), "--directory", str(args.model_path),
                    "--inventory", str(args.inventory)], check=True)
    for filename, digest in manifest["tokenizer_hashes"].items():
        if hashlib.sha256((args.model_path / filename).read_bytes()).hexdigest() != digest:
            raise SystemExit("training tokenizer differs: " + filename)
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, PeftModel
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("BF16-capable CUDA required")
    set_seed(61001)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
                                               trust_remote_code=False, dtype=torch.bfloat16,
                                               attn_implementation="sdpa")
    base.config.use_cache = False
    policy = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                            "gate_proj", "up_proj", "down_proj"],
                                            task_type="CAUSAL_LM", bias="none"))
    def collate(features):
        return {k: torch.tensor(v, dtype=torch.long) for k, v in
                pad_training_batch(features, tokenizer.pad_token_id).items()}
    fields = ("input_ids", "attention_mask", "labels")
    if args.formal_one_epoch:
        if args.max_smoke_tokens is not None:
            parser.error("--max-smoke-tokens is only valid with --train")
        smoke_row = min(rows, key=lambda row: len(row["input_ids"]))
        dataset = [{k: row[k] for k in fields} for row in rows]
    else:
        # The cap selects a complete real trajectory; it never truncates observations or labels.
        smoke_row = select_smoke_row(rows, args.max_smoke_tokens)
        dataset = [{k: smoke_row[k] for k in fields}]
    training_options = ({"num_train_epochs": 1.0} if args.formal_one_epoch else {"max_steps": 1})
    config = TrainingArguments(output_dir=str(args.output),
                               per_device_train_batch_size=1, gradient_accumulation_steps=1,
                               learning_rate=1e-5, bf16=True, gradient_checkpointing=True,
                               gradient_checkpointing_kwargs={"use_reentrant": False},
                               use_liger_kernel=True,
                               optim="adamw_torch", save_strategy="no", report_to=[],
                               logging_steps=1, seed=61001, data_seed=61001,
                               dataloader_num_workers=0, **training_options)
    trainer = Trainer(model=policy, args=config, train_dataset=dataset,
                      data_collator=collate, processing_class=tokenizer)
    original_adapter = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(policy).items()}
    torch.cuda.reset_peak_memory_stats()
    trained = trainer.train()
    if not math.isfinite(trained.training_loss):
        raise RuntimeError("nonfinite training loss")
    updated_adapter = get_peft_model_state_dict(policy)
    if not all(torch.isfinite(v).all().item() for v in updated_adapter.values()):
        raise RuntimeError("nonfinite adapter weights")
    changed = sum(not torch.equal(original_adapter[k], v.detach().cpu()) for k, v in updated_adapter.items())
    if not changed:
        raise RuntimeError("optimizer step did not change adapter weights")
    saved_adapter = {k: v.detach().cpu().clone() for k, v in updated_adapter.items()}
    adapter = args.output / "adapter"
    policy.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    if args.formal_one_epoch:
        losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
        if len(losses) != len(rows) or not all(math.isfinite(value) for value in losses):
            raise RuntimeError("formal run does not have one finite logged loss per trajectory")
        adapter_file = adapter / "adapter_model.safetensors"
        summary.update(mode="formal_one_epoch_sft", training_performed=True,
                       runtime_checked=True, training_loss=trained.training_loss,
                       first_step_loss=losses[0], final_step_loss=losses[-1],
                       min_step_loss=min(losses), max_step_loss=max(losses),
                       changed_adapter_tensors=changed,
                       adapter_sha256=hashlib.sha256(adapter_file.read_bytes()).hexdigest(),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       transformers_version=transformers.__version__, peft_version=peft.__version__,
                       torch_version=torch.__version__,
                       liger_kernel_version=importlib.metadata.version("liger-kernel"),
                       formal_sft_performed=True, epochs=1, optimizer_steps=len(rows),
                       post_train_reload_check="covered_by_one_step_gpu_smoke")
        (args.output / "formal-sft-report.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return
    policy.eval()
    batch = {k: v.to(policy.device) for k, v in collate(dataset).items()}
    # Trainer wraps the original forward in autocast; the reloaded model does not.
    # Use the same explicit precision context on both sides of the comparison.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        before = policy(**batch, skip_logits=True).loss.detach().float().item()
    # Reload the saved adapter onto the same frozen base; never merge LoRA weights.
    unloaded = policy.unload()
    if hasattr(unloaded, "peft_config"):
        del unloaded.peft_config
    restored = PeftModel.from_pretrained(unloaded, adapter, is_trainable=False)
    restored_adapter = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(restored).items()}
    if saved_adapter.keys() != restored_adapter.keys() or any(
            not torch.equal(value, restored_adapter[key]) for key, value in saved_adapter.items()):
        raise RuntimeError("reloaded adapter tensors differ from saved tensors")
    restored.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        after = restored(**batch, skip_logits=True).loss.detach().float().item()
    if not math.isfinite(before) or not math.isfinite(after) or not math.isclose(before, after, rel_tol=1e-4, abs_tol=1e-4):
        raise RuntimeError(f"adapter reload loss mismatch: {before} vs {after}")
    summary.update(mode=("formal_one_epoch_sft" if args.formal_one_epoch else "one_step_gpu_smoke"),
                   training_performed=True, runtime_checked=True,
                   training_loss=trained.training_loss, loss_before_reload=before,
                   loss_after_reload=after, smoke_tokens=len(smoke_row["input_ids"]),
                   smoke_token_cap=args.max_smoke_tokens,
                   changed_adapter_tensors=changed,
                   adapter_state_reloaded_exactly=True,
                   reload_loss_autocast="bfloat16", reload_loss_tolerance=1e-4,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   transformers_version=transformers.__version__, peft_version=peft.__version__,
                   torch_version=torch.__version__,
                   liger_kernel_version=importlib.metadata.version("liger-kernel"),
                   formal_sft_performed=args.formal_one_epoch,
                   epochs=(1 if args.formal_one_epoch else None),
                   optimizer_steps=(len(rows) if args.formal_one_epoch else 1))
    report_name = "formal-sft-report.json" if args.formal_one_epoch else "smoke-report.json"
    (args.output / report_name).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
