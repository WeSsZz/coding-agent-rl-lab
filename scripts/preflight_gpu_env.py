"""Pre-flight a rented GPU box before any training is attempted.

Runs on the GPU machine and answers only questions that can invalidate the whole plan. It
spends no GPU time on training and downloads no model weights. Every check prints `ok`,
`WARN` or `FAIL`, and the exit code is non-zero only when a FAIL means the pinned stack cannot
work here at all.

Checks, and why each one is here:

* **Driver CUDA version vs the cu130 stack.** veRL v0.9.1's wheels are cu130. A driver that
  cannot do CUDA 13 is a blocking incompatibility, and it is far cheaper to find that out now
  than after an hour of `uv sync`.
* **GPU name and total VRAM.** The plan assumes one ~96 GB card. A different card changes how
  much of the LoRA + vLLM + reference-model budget is available, so the real number is recorded
  rather than assumed.
* **CPU count, RAM, free disk.** The rollout farm lives on this box, not on the worker VM: each
  task container is capped at 2 GB and the worker has 5 GB in total. Disk has to hold the model,
  the checkpoints and the pulled task images.
* **Reachability of hf-mirror.com.** `huggingface.co` is unreachable from this project's
  networks; if the rental box is the same, weights must be uploaded rather than downloaded.
* **Registered packages.** veRL, vLLM, torch, transformers, peft, flash-attn and their versions,
  so the run record can state what actually ran instead of what was intended.

Usage (on the GPU box):
    python scripts/preflight_gpu_env.py [--json report.json]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

OK = "ok"
WARN = "WARN"
FAIL = "FAIL"

REQUIRED_CUDA_MAJOR = 13
MIN_FREE_DISK_GB = 80.0


def _run(command: list[str], timeout: int = 30) -> tuple[int, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, f"{type(exc).__name__}: {exc}"
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def check_nvidia_smi() -> dict[str, Any]:
    code, output = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if code != 0:
        return {"status": FAIL, "detail": f"nvidia-smi unavailable: {output.strip()[:200]}"}
    gpus = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            gpus.append(
                {
                    "name": parts[0],
                    "memory_total_mib": int(float(parts[1])),
                    "driver_version": parts[2],
                }
            )
    if not gpus:
        return {"status": FAIL, "detail": "nvidia-smi returned no devices"}
    total_gib = [round(gpu["memory_total_mib"] / 1024, 1) for gpu in gpus]
    status = OK if len(gpus) == 1 and total_gib[0] >= 80 else WARN
    return {"status": status, "gpus": gpus, "total_gib": total_gib, "count": len(gpus)}


def check_driver_cuda() -> dict[str, Any]:
    code, output = _run(["nvidia-smi"])
    if code != 0:
        return {"status": FAIL, "detail": "nvidia-smi unavailable"}
    version = None
    for line in output.splitlines():
        if "CUDA Version" in line:
            tail = line.split("CUDA Version:")[-1].strip()
            version = tail.split()[0] if tail else None
            break
    if version is None:
        return {"status": WARN, "detail": "could not read the driver's CUDA version from nvidia-smi"}
    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError):
        return {"status": WARN, "detail": f"unparseable CUDA version {version!r}"}
    if major >= REQUIRED_CUDA_MAJOR:
        return {"status": OK, "driver_cuda": version}
    return {
        "status": FAIL,
        "driver_cuda": version,
        "detail": (
            f"driver reports CUDA {version} but the pinned stack is cu130; "
            "update the driver or choose a different veRL tag"
        ),
    }


def check_host() -> dict[str, Any]:
    cpu = os.cpu_count() or 0
    memory_gib = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("MemTotal:"):
                    memory_gib = round(int(line.split()[1]) / 1024 / 1024, 1)
                    break
    except OSError:
        pass
    disk_free_gib = None
    try:
        usage = shutil.disk_usage("/")
        disk_free_gib = round(usage.free / 1024**3, 1)
    except OSError:
        pass
    status = OK
    detail = ""
    if memory_gib is not None and memory_gib < 32:
        status = WARN
        detail = "under 32 GiB RAM: the rollout farm runs many sandboxes on this box"
    if disk_free_gib is not None and disk_free_gib < MIN_FREE_DISK_GB:
        status = FAIL
        detail = f"under {MIN_FREE_DISK_GB} GiB free disk; the model, checkpoints and images need more"
    return {
        "status": status,
        "cpu_count": cpu,
        "memory_gib": memory_gib,
        "disk_free_gib": disk_free_gib,
        "python": platform.python_version(),
        "detail": detail,
    }


def check_hf_reachability() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for host, url in (
        ("huggingface.co", "https://huggingface.co/"),
        ("hf-mirror.com", "https://hf-mirror.com/"),
    ):
        code, _ = _run(
            ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "15", url],
            timeout=25,
        )
        results[host] = code == 0
    if results.get("hf-mirror.com"):
        status = OK
        detail = ""
    elif results.get("huggingface.co"):
        status = OK
        detail = "direct HuggingFace works; hf-mirror does not, which is fine"
    else:
        status = WARN
        detail = "neither host is reachable: upload the model weights instead of downloading them"
    return {"status": status, "reachable": results, "detail": detail}


def check_packages() -> dict[str, Any]:
    wanted = ("torch", "transformers", "vllm", "ray", "peft", "flash_attn", "tensordict", "verl")
    versions: dict[str, str | None] = {}
    for name in wanted:
        code, output = _run(
            [
                sys.executable,
                "-c",
                f"import importlib.metadata as m; print(m.version('{name}'))",
            ]
        )
        versions[name] = output.strip() if code == 0 else None
    missing = [name for name, version in versions.items() if version is None]
    status = OK if not missing else WARN
    # `verl` and `flash_attn` are the two whose absence is expected before installation.
    return {
        "status": status,
        "versions": versions,
        "missing": missing,
        "executable": sys.executable,
    }


def check_torch_cuda() -> dict[str, Any]:
    code, output = _run(
        [
            sys.executable,
            "-c",
            "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), "
            "torch.cuda.device_count())",
        ],
        timeout=120,
    )
    if code != 0:
        return {"status": WARN, "detail": f"torch not importable: {output.strip()[:200]}"}
    parts = output.strip().split()
    if len(parts) >= 4:
        return {
            "status": OK if parts[2] == "True" else FAIL,
            "torch": parts[0],
            "torch_cuda": parts[1],
            "cuda_available": parts[2] == "True",
            "device_count": int(parts[3]),
        }
    return {"status": WARN, "detail": output.strip()[:200]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the structured report here")
    arguments = parser.parse_args(argv)

    checks = {
        "gpu": check_nvidia_smi(),
        "driver_cuda": check_driver_cuda(),
        "host": check_host(),
        "hf": check_hf_reachability(),
        "packages": check_packages(),
        "torch": check_torch_cuda(),
    }

    for name, result in checks.items():
        status = result.get("status", WARN)
        detail = result.get("detail", "")
        print(f"{status:4s}  {name:12s} {json.dumps({k: v for k, v in result.items() if k != 'status'})[:300]}")
        if detail:
            print(f"        {detail}")

    hard_failures = [name for name, result in checks.items() if result.get("status") == FAIL]
    print()
    if hard_failures:
        print(f"RESULT: FAILED ({', '.join(hard_failures)})")
    else:
        warnings = [name for name, result in checks.items() if result.get("status") == WARN]
        print(f"RESULT: OK{' with warnings (' + ', '.join(warnings) + ')' if warnings else ''}")

    report = {"schema_version": 1, "checks": checks, "hard_failures": hard_failures}
    if arguments.json:
        arguments.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {arguments.json}")
    return 1 if hard_failures else 0


if __name__ == "__main__":
    sys.exit(main())
