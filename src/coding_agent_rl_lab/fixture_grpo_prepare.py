from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .evaluation import load_builtin_tasks
from .grpo_environment import build_grpo_prompt_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write answer-free GRPO prompts for trusted fixtures")
    parser.add_argument("--output", default="work/fixture-grpo-prompts.jsonl")
    parser.add_argument("--max-steps", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    project_root = Path(__file__).resolve().parents[2]
    tasks = tuple(
        replace(task, max_steps=args.max_steps)
        for task in load_builtin_tasks(project_root)
    )
    rows = build_grpo_prompt_rows(tasks)
    output = project_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"READY tasks={len(rows)} prompts={output}")


if __name__ == "__main__":
    main()
