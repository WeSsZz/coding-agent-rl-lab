#!/usr/bin/env bash
# Fetch the pinned upstream inventory; no weights are downloaded.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PINNED_MODEL=$(python3 "$ROOT/scripts/model_contract.py" --field id)
PINNED_REVISION=$(python3 "$ROOT/scripts/model_contract.py" --field revision)
MODEL=${1:-$PINNED_MODEL}
REVISION=${2:-$PINNED_REVISION}
ENDPOINT=${3:-https://hf-mirror.com}
[ "$MODEL" = "$PINNED_MODEL" ] && [ "$REVISION" = "$PINNED_REVISION" ] || {
  echo "Model/revision differs from route-v2 manifest" >&2; exit 2;
}
mkdir -p "$ROOT/work"
TARGET="$ROOT/work/model-inventory.json"
curl --fail --location --retry 2 --max-time 90 \
  "$ENDPOINT/api/models/$MODEL/revision/$REVISION?blobs=true" -o "$TARGET.part"
python3 - "$TARGET.part" "$MODEL" "$REVISION" <<'PYCODE'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    value = json.load(stream)
if value.get("id") != sys.argv[2] or value.get("sha") != sys.argv[3]:
    raise SystemExit("Pinned inventory identity mismatch")
if not value.get("siblings"):
    raise SystemExit("Inventory has no files")
PYCODE
mv -- "$TARGET.part" "$TARGET"
echo "Pinned inventory saved: $TARGET; source=$ENDPOINT"
