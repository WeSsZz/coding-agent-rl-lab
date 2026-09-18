from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a deterministic SHA-256 manifest for a directory")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite manifest: {output}")
    if not root.is_dir() or root not in output.parents:
        raise SystemExit("output must be a new file inside the manifest root")
    lines: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == output:
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"files={len(lines)} output={output}")


if __name__ == "__main__":
    main()
