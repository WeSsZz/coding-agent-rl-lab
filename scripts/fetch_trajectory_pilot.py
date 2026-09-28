"""Fetch a bounded, deterministic format pilot; not a representative training set.

Requires pyarrow, pandas, fsspec and aiohttp. A row group is read, although only
the first 32 rows are retained. No trajectory content is executed.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


DATASET = "nebius/SWE-rebench-openhands-trajectories"
REVISION = "35455389ab51bf5e2306bfd436ef72d0f98bf882"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--short-successes", type=int, default=0,
                        help="Shortlist this many unique successful issues from row group 0, by text size; tokenizer validation remains required")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output already exists")
    if not 0 <= args.short_successes <= 256:
        raise SystemExit("short-successes must be between 0 and 256")
    import aiohttp
    import fsspec
    import pyarrow.parquet as pq
    endpoint = "https://hf-mirror.com"
    api = f"{endpoint}/api/datasets/{DATASET}/revision/{REVISION}?blobs=true"
    inventory_bytes = urlopen(api, timeout=60).read()
    inventory = json.loads(inventory_bytes)
    if inventory["id"] != DATASET or inventory["sha"] != REVISION:
        raise SystemExit("dataset revision mismatch")
    base = f"{endpoint}/datasets/{DATASET}/resolve/{REVISION}/"
    args.output.mkdir(parents=True)
    (args.output / "inventory.json").write_bytes(inventory_bytes)
    for filename in ("README.md", "LICENSE"):
        data = urlopen(base + filename, timeout=60).read()
        entry = next(e for e in inventory["siblings"] if e["rfilename"] == filename)
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if digest != entry["blobId"]:
            raise SystemExit(f"hash mismatch: {filename}")
        (args.output / filename).write_bytes(data)
    with fsspec.open(base + "trajectories.parquet", "rb", block_size=65536,
                     client_kwargs={"timeout": aiohttp.ClientTimeout(total=600), "trust_env": True}) as stream:
        parquet = pq.ParquetFile(stream)
        group = parquet.metadata.row_group(0)
        compressed = sum(group.column(i).total_compressed_size for i in range(group.num_columns))
        if compressed > 256 * 1024**2 or group.total_byte_size > 1024**3:
            raise SystemExit("first row group exceeds pilot budget")
        table = parquet.read_row_group(0)
        selection = None
        if args.short_successes:
            # Size is only a cheap candidate ordering. Never truncate messages.
            rows = table.to_pylist()
            candidates = [(i, r) for i, r in enumerate(rows)
                          if r.get("resolved") == 1 and r.get("exit_status") == "submit"]
            candidates.sort(key=lambda item: (len(json.dumps(item[1]["trajectory"])), item[1]["trajectory_id"]))
            indices, seen = [], set()
            for index, row in candidates:
                if row["instance_id"] in seen:
                    continue
                seen.add(row["instance_id"])
                indices.append(index)
                if len(indices) >= args.short_successes:
                    break
            table = table.take(indices)
            selection = {"rule": "shortest serialized trajectory per unique instance; first row group only",
                         "row_group_rows": group.num_rows, "success_candidates": len(candidates),
                         "source_row_indices": indices, "representative_sample": False}
        else:
            table = table.slice(0, 32)
        table.to_pandas().to_json(args.output / "sample.jsonl", orient="records", lines=True, force_ascii=True)
    report = {"rows": table.num_rows, "row_group_compressed_bytes": compressed,
                      "revision": REVISION, "whole_parquet_hash_verified": False,
                      "selection": selection,
                      "sample_sha256": hashlib.sha256((args.output / "sample.jsonl").read_bytes()).hexdigest()}
    (args.output / "fetch-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "selection"}))


if __name__ == "__main__":
    main()
