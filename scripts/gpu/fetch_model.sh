#!/usr/bin/env bash
# Fetch only the model/revision selected in route-v2-model.json, then verify every weight shard.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PINNED_MODEL=$(python3 "$ROOT/scripts/model_contract.py" --field id)
PINNED_REVISION=$(python3 "$ROOT/scripts/model_contract.py" --field revision)
MODEL=${MODEL_ID:-$PINNED_MODEL}
REVISION=${REVISION:-$PINNED_REVISION}
[ "$MODEL" = "$PINNED_MODEL" ] && [ "$REVISION" = "$PINNED_REVISION" ] || {
  echo "Download model/revision differs from manifest" >&2; exit 2;
}
TARGET=${1:-/root/autodl-tmp/models/${MODEL##*/}}
mkdir -p "$TARGET"

echo "=== target ==="
echo "  $TARGET"

echo
echo "=== choosing an endpoint ==="
ENDPOINT=""
probe_endpoint() {
  # The mirror is intermittently unreachable: the same URL that answers 200 can drop the
  # connection a minute later. Retry before concluding anything.
  local candidate=$1 attempt code
  for attempt in 1 2 3; do
    code=$(curl -sL -o /tmp/model-api.json -w '%{http_code}' --max-time 20 \
           "$candidate/api/models/$MODEL" 2>/dev/null) || code=000
    if [ "$code" = "200" ]; then
      printf '%s' "$code"
      return 0
    fi
    sleep 3
  done
  printf '%s' "${code:-000}"
  return 1
}

for candidate in "https://hf-mirror.com" "https://huggingface.co"; do
  printf '  %-28s ' "$candidate"
  code=$(probe_endpoint "$candidate")
  echo "$code"
  if [ "$code" = "200" ] && [ -z "$ENDPOINT" ]; then
    ENDPOINT="$candidate"

  fi
done

if [ -z "$ENDPOINT" ]; then
  echo "No endpoint available; stop without downloading an unpinned alternative" >&2
  exit 1
fi
bash "$ROOT/scripts/gpu/verify_model_revision.sh" "$MODEL" "$REVISION" "$ENDPOINT" || exit 1

echo
echo "=== attempting a download client ==="
START=$(date +%s)
CLIENT=""
TRIED_HUB=0

if command -v hf >/dev/null 2>&1; then
  CLIENT="hf"
  HF_ENDPOINT="$ENDPOINT" hf download "$MODEL" ${REVISION:+--revision "$REVISION"} --local-dir "$TARGET"
elif command -v huggingface-cli >/dev/null 2>&1; then
  CLIENT="huggingface-cli"
  HF_ENDPOINT="$ENDPOINT" huggingface-cli download "$MODEL" ${REVISION:+--revision "$REVISION"} --local-dir "$TARGET"
elif command -v uvx >/dev/null 2>&1; then
  CLIENT="uvx"
  HF_ENDPOINT="$ENDPOINT" uvx --from huggingface_hub hf download "$MODEL" ${REVISION:+--revision "$REVISION"} --local-dir "$TARGET"
else
  TRIED_HUB=1
  python3 - "$ENDPOINT" "$MODEL" "$REVISION" "$TARGET" <<'PY' && CLIENT="huggingface_hub"
import os, sys
try:
    from huggingface_hub import snapshot_download
except Exception:
    sys.exit(3)
endpoint, model, revision, target = sys.argv[1:5]
os.environ["HF_ENDPOINT"] = endpoint
snapshot_download(repo_id=model, revision=revision or None, local_dir=target)
PY
  if [ $? -ne 0 ]; then
    echo "  huggingface_hub not importable; falling back to curl"
    CLIENT=""
  fi
fi

if [ -z "$CLIENT" ]; then
  echo "  using the dependency-free curl path"
  python3 - "$ENDPOINT" "$MODEL" "$REVISION" "$TARGET" "$ROOT/work/model-inventory.json" <<'PY'
import json, os, subprocess, sys

endpoint, model, revision, target, cached_listing = sys.argv[1:6]
revision = revision or "main"


def listing_from_cache():
    """The endpoint probe already fetched this; reusing it avoids a second flaky request."""
    try:
        return json.loads(open(cached_listing, encoding="utf-8").read())
    except Exception:
        return None


def listing_fresh(attempts=3):
    last = ""
    for _ in range(attempts):
        result = subprocess.run(
            ["curl", "-sL", "--fail", "--max-time", "60", f"{endpoint}/api/models/{model}/revision/{revision}?blobs=true"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            try:
                return json.loads(result.stdout)
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
                continue
        last = f"curl exit {result.returncode} {result.stderr.strip()[:120]}"
    print(f"  FAILED to list files: {last}")
    return None


data = listing_from_cache() or listing_fresh()
if data is None:
    sys.exit(1)

siblings = [entry["rfilename"] for entry in data.get("siblings", [])]
siblings = [name for name in siblings if name.endswith((".safetensors", ".json", ".txt", ".model", ".jinja"))]
if not siblings:
    print("  FAILED: the model API returned no files")
    sys.exit(1)

print(f"  {len(siblings)} file(s) listed")


def fetch(name, attempts=3):
    destination = os.path.join(target, name)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    url = f"{endpoint}/{model}/resolve/{revision}/{name}"
    for attempt in range(1, attempts + 1):
        result = subprocess.run(
            ["curl", "-sL", "--fail", "--max-time", "3600", "-o", destination,
             "-w", "%{size_download}", url],
            capture_output=True, text=True,
        )
        size = result.stdout.strip()
        if result.returncode == 0 and size not in ("", "0"):
            return True, f"{size} bytes"
        if attempt < attempts:
            continue
        return False, f"curl exit {result.returncode}, {size or 'no'} bytes"
    return False, "unreachable"


failed = 0
for index, name in enumerate(siblings, 1):
    ok, detail = fetch(name)
    print(f"    [{index}/{len(siblings)}] {name} {'ok' if ok else 'FAILED'} ({detail})")
    if not ok:
        failed += 1
sys.exit(1 if failed else 0)
PY
fi
END=$(date +%s)
echo "  client: ${CLIENT:-curl}; elapsed $((END - START))s"

echo
echo "=== verifying complete pinned model contents ==="
python3 "$ROOT/scripts/model_contract.py" --directory "$TARGET" \
  --inventory "$ROOT/work/model-inventory.json" --output "$TARGET/route-v2-verification.json"
