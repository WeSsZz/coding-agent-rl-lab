"""Pin a HuggingFace chat template into this repository as a reviewable artefact.

Phase 3 requires the tokenizer and chat template to be fixed before any GPU time is spent. A
template that is only reachable by downloading a model is not fixed; this writes the exact
template text plus the provenance needed to re-derive it, so a diff of the template is a diff
of the contract between the SFT data, the RL rollout and the evaluation.

Usage:
    python scripts/pin_qwen_chat_template.py \
        --tokenizer-config work/private/qwen3-8b-tokenizer_config.json \
        --model-id Qwen/Qwen3-8B \
        --output envs/qwen3-8b-chat-template.jinja
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PinError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = json.loads(arguments.tokenizer_config.read_text(encoding="utf-8"))
    template = config.get("chat_template")
    if not isinstance(template, str) or not template.strip():
        print("FAILED  tokenizer config has no chat_template")
        return 1

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    # Write bytes, not text: `Path.write_text` translates "\n" to the platform line separator,
    # so on Windows the file on disk would not be the string that was hashed. A pinned artefact
    # whose recorded hash does not match its own bytes is worse than no hash at all.
    arguments.output.write_bytes(template.encode("utf-8"))

    manifest_path = arguments.manifest or arguments.output.with_suffix(".manifest.json")
    # Required: without these the adopted OpenHands function-calling schema cannot be rendered at
    # all. Optional and merely recorded: Qwen3 templates carry an `enable_thinking` switch and
    # Qwen2.5-Coder templates do not, and that difference is expected rather than a defect.
    required_markers = {
        "tools_branch": "{%- if tools %}" in template or "{% if tools %}" in template,
        "tool_call": "<tool_call>" in template,
        "tool_response": "<tool_response>" in template,
    }
    optional_markers = {
        "enable_thinking": "enable_thinking" in template,
        # Whether a JSON *string* of arguments is passed through or run through `tojson` again.
        # Qwen3 guards this with `is string`; Qwen2.5-Coder does not, so a trajectory whose
        # arguments are stored as a string must be deserialized before rendering or the tool call
        # ends up double-encoded.
        "arguments_string_passthrough": "tool_call.arguments is string" in template,
        "reasoning_content": "reasoning_content" in template,
    }

    manifest = {
        "schema_version": 1,
        "manifest_id": "chat-template",
        "generated_at": "2026-09-22",
        "model_id": arguments.model_id,
        "revision": arguments.revision,
        "source": {
            "file": "tokenizer_config.json",
            "local_copy_sha256": sha256_of(arguments.tokenizer_config),
            "retrieved_via": "https://hf-mirror.com (huggingface.co is unreachable from this project)",
        },
        "tokenizer_class": config.get("tokenizer_class"),
        "model_max_length": config.get("model_max_length"),
        "template": {
            "path": str(arguments.output).replace("\\", "/"),
            "sha256": sha256_of(arguments.output),
            "characters": len(template),
            "line_endings": "LF",
        },
        "markers_required": required_markers,
        "markers_optional": optional_markers,
        "non_thinking_contract": (
            "Qwen3 templates emit '<think>\\n\\n</think>\\n\\n' when `enable_thinking` is defined "
            "and false; that empty block is what non-thinking mode means for those models, and SFT "
            "rendering and RL rollout must pass the same flag or the token streams differ. This "
            "template has no such switch, so the generation prompt ends at the assistant header."
            if not optional_markers["enable_thinking"]
            else "The template emits '<think>\\n\\n</think>\\n\\n' when `enable_thinking` is defined "
            "and false. That empty block is what non-thinking mode means for this project, and both "
            "SFT rendering and RL rollout must pass the same flag or the token streams differ."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {arguments.output} ({len(template)} chars)")
    print(f"  template sha256 {manifest['template']['sha256']}")
    print(f"  source   sha256 {manifest['source']['local_copy_sha256']}")
    print(f"  optional markers: " + ", ".join(f"{k}={'Y' if v else 'N'}" for k, v in optional_markers.items()))
    print(f"wrote {manifest_path}")

    missing = [name for name, present in required_markers.items() if not present]
    if missing:
        print(f"FAILED  template is missing required markers: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
