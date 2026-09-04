"""Isolate the GRPO training environment from an incompatible system vLLM.

AutoDL's inference image exposes vLLM 0.11 through system site-packages.  Newer
TRL releases import vLLM whenever the package is discoverable, even when
``use_vllm`` is disabled, and require a newer vLLM API.  The GRPO virtual
environment deliberately reuses the system PyTorch/CUDA installation, so hide
only the incompatible ``vllm`` package there.  The separately running vLLM
server uses the base environment and is unaffected.
"""

from __future__ import annotations

import importlib.util
from importlib.machinery import ModuleSpec
from typing import Callable


_original_find_spec: Callable[..., ModuleSpec | None] = importlib.util.find_spec


def _find_spec_without_system_vllm(name: str, package: str | None = None) -> ModuleSpec | None:
    if name == "vllm" or name.startswith("vllm."):
        return None
    return _original_find_spec(name, package)


importlib.util.find_spec = _find_spec_without_system_vllm
