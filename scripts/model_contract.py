"""Single model selection and fail-closed local snapshot verification; no GPU required."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/route-v2-model.json"


def selected_model(manifest: Path = MANIFEST) -> dict:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    model = data["models"]["mainline"]
    if model["id"] != data["decision"]["mainline"]:
        raise ValueError("model decision and mainline disagree")
    return model


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def validate_snapshot(directory: Path, model: dict, inventory: dict | None = None) -> dict:
    directory = directory.resolve()
    for name, key in (("config.json", "config_json_sha256"),
                      ("tokenizer_config.json", "tokenizer_config_json_sha256")):
        if digest(directory / name) != model[key]:
            raise ValueError(f"pinned {name} mismatch")
    if not (directory / "tokenizer.json").is_file():
        raise ValueError("tokenizer.json missing")
    index_path = directory / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    if not weight_map:
        raise ValueError("empty safetensors weight map")
    shards = sorted(set(weight_map.values()))
    for name in shards:
        if Path(name).name != name or "\\" in name:
            raise ValueError("unsafe shard name")
        path = directory / name
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"truncated shard: {name}")
            length = struct.unpack("<Q", prefix)[0]
            if length > 100 * 1024 * 1024:
                raise ValueError(f"invalid safetensors header: {name}")
            header = json.loads(stream.read(length))
        data_size = path.stat().st_size - 8 - length
        offsets = [v["data_offsets"] for k, v in header.items() if k != "__metadata__"]
        if not offsets or any(not 0 <= a <= b <= data_size for a, b in offsets):
            raise ValueError(f"invalid/truncated tensor data: {name}")
        if max(b for _, b in offsets) != data_size:
            raise ValueError(f"unexpected shard size: {name}")
        if any(key not in header for key, file in weight_map.items() if file == name):
            raise ValueError(f"tensor missing from shard: {name}")
    # Config agreement and complete shards alone cannot prove weight provenance.
    if inventory is None:
        raise ValueError("pinned upstream file inventory required to verify weight identity")
    if inventory.get("sha") != model["revision"] or inventory.get("id") != model["id"]:
        raise ValueError("upstream inventory model/revision mismatch")
    entries = {item["rfilename"]: item for item in inventory["siblings"]}
    checked = {}
    for name in [*shards, "tokenizer.json", "model.safetensors.index.json"]:
        entry = entries[name]
        path = directory / name
        if entry.get("lfs"):
            expected = entry["lfs"]["sha256"]
            actual = digest(path)
        else:
            raw = path.read_bytes()
            actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            expected = entry["blobId"]
        if actual != expected:
            raise ValueError(f"upstream content hash mismatch: {name}")
        checked[name] = actual
    return {"model": model["id"], "revision": model["revision"],
            "path": str(directory), "verified": True, "files": checked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--field", choices=["id", "revision", "chat_template"])
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = selected_model(args.manifest)
    if args.field:
        print(model[args.field])
        return
    if args.directory is None:
        parser.error("--directory or --field required")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8")) if args.inventory else None
    try:
        report = validate_snapshot(args.directory, model, inventory)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"MODEL VERIFICATION FAILED: {exc}\n")
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
