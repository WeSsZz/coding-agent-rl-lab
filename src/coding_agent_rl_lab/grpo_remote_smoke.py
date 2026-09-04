from __future__ import annotations

import argparse
import json
from pathlib import Path

from .grpo_remote import RemoteGRPOCodingEnvironment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test the loopback GRPO worker")
    parser.add_argument("--base-url", default="http://127.0.0.1:9010")
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--task-id", default="getmoto__moto-7365")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    token = Path(args.token_file).read_text(encoding="utf-8").strip()
    environment = RemoteGRPOCodingEnvironment(args.base_url, token)
    initial = environment.reset(task_id=args.task_id)
    files = environment.list_files()
    reward = environment.reward
    print(
        json.dumps(
            {
                "task_id": args.task_id,
                "baseline_failed": "Tests failed" in initial,
                "file_listing_received": bool(files.strip()),
                "reward": reward,
                "code_modified": False,
                "training_performed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
