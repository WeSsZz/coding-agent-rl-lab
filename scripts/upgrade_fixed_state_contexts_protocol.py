from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from coding_agent_rl_lab.sft_train import _structured_grpo_history_message


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Version fixed-state contexts with rollout-equivalent tool-call history"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    source = Path(args.input)
    output = Path(args.output)
    report_path = Path(args.report)
    if output.exists() or report_path.exists():
        raise SystemExit("refusing to overwrite upgraded context evidence")

    rows = _jsonl(source)
    assistant_count = 0
    for row in rows:
        upgraded = []
        for message in row["prompt"]:
            converted = _structured_grpo_history_message(message)
            if converted.get("role") == "assistant" and "tool_calls" in converted:
                assistant_count += 1
            upgraded.append(converted)
        row["prompt"] = upgraded
        row["protocol_version"] = "tools-structured-history-v2"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "completed": True,
        "source_path": str(source.resolve()),
        "source_sha256": _sha256(source),
        "output_path": str(output.resolve()),
        "output_sha256": _sha256(output),
        "context_count": len(rows),
        "structured_assistant_history_count": assistant_count,
        "state_ids": [row["state_id"] for row in rows],
        "protocol_version": "tools-structured-history-v2",
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if __name__ == "__main__":
    main()
