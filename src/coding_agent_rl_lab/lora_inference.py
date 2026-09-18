from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LoRAInferenceError(RuntimeError):
    pass


def merge_lora_adapter_for_inference(model: Any, adapter_path: Path) -> dict[str, Any]:
    """Merge a simple PEFT LoRA adapter into an in-memory base model.

    This deliberately supports only the all-linear, no-bias, non-DoRA adapters
    produced by this project. The model and adapter files on disk are unchanged.
    """

    import torch
    from safetensors import safe_open

    config_path = adapter_path / "adapter_config.json"
    weights_path = adapter_path / "adapter_model.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA":
        raise LoRAInferenceError("only PEFT LORA adapters are supported")
    if config.get("bias", "none") != "none":
        raise LoRAInferenceError("LoRA inference merge refuses adapter bias")
    if config.get("use_dora", False):
        raise LoRAInferenceError("LoRA inference merge refuses DoRA adapters")
    if config.get("fan_in_fan_out", False):
        raise LoRAInferenceError("LoRA inference merge refuses fan_in_fan_out")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise LoRAInferenceError("LoRA inference merge refuses per-module rank/alpha")
    rank = config.get("r")
    alpha = config.get("lora_alpha")
    if not isinstance(rank, int) or rank <= 0 or not isinstance(alpha, (int, float)):
        raise LoRAInferenceError("invalid LoRA rank or alpha")
    scaling = float(alpha) / rank

    parameters = dict(model.named_parameters())
    with safe_open(weights_path, framework="pt", device="cpu") as stream:
        keys = set(stream.keys())
        a_keys = sorted(key for key in keys if key.endswith(".lora_A.weight"))
        b_keys = {key for key in keys if key.endswith(".lora_B.weight")}
        expected = set(a_keys) | {
            key[: -len(".lora_A.weight")] + ".lora_B.weight" for key in a_keys
        }
        unexpected = sorted(keys - expected)
        if unexpected:
            raise LoRAInferenceError(f"unsupported adapter tensor keys: {unexpected[:3]}")
        if not a_keys:
            raise LoRAInferenceError("adapter contains no LoRA A tensors")

        merged: list[str] = []
        with torch.no_grad():
            for a_key in a_keys:
                prefix = a_key[: -len(".lora_A.weight")]
                b_key = prefix + ".lora_B.weight"
                if b_key not in b_keys:
                    raise LoRAInferenceError(f"missing LoRA B tensor for {a_key}")
                parameter_name = _base_parameter_name(prefix, parameters)
                parameter = parameters[parameter_name]
                a = stream.get_tensor(a_key).to(device=parameter.device, dtype=torch.float32)
                b = stream.get_tensor(b_key).to(device=parameter.device, dtype=torch.float32)
                if a.shape[0] != rank or b.shape[1] != rank:
                    raise LoRAInferenceError(f"rank mismatch for {prefix}")
                delta = torch.matmul(b, a).mul_(scaling)
                if tuple(delta.shape) != tuple(parameter.shape):
                    raise LoRAInferenceError(
                        f"shape mismatch for {parameter_name}: {tuple(delta.shape)} != "
                        f"{tuple(parameter.shape)}"
                    )
                parameter.add_(delta.to(dtype=parameter.dtype))
                merged.append(parameter_name)
                del a, b, delta

    return {
        "merge_kind": "in-memory-standard-lora",
        "rank": rank,
        "alpha": float(alpha),
        "scaling": scaling,
        "merged_parameter_count": len(merged),
        "merged_parameters": merged,
        "disk_weights_modified": False,
    }


def _base_parameter_name(prefix: str, parameters: dict[str, Any]) -> str:
    candidates = [prefix + ".weight"]
    for leading in ("base_model.model.", "base_model."):
        if prefix.startswith(leading):
            candidates.append(prefix[len(leading) :] + ".weight")
    for candidate in candidates:
        if candidate in parameters:
            return candidate
    raise LoRAInferenceError(f"no base parameter for adapter tensor prefix: {prefix}")
