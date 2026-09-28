#!/usr/bin/env bash
# GPU-only entry for the pinned Qwen3-8B complete-trajectory pilot.
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
MODE=${1:-}
case "$MODE" in
  smoke-24k) TRAIN_ARGS=(--train --max-smoke-tokens 24576) ;;
  smoke-32k) TRAIN_ARGS=(--train) ;;
  formal) TRAIN_ARGS=(--formal-one-epoch) ;;
  *) echo "Usage: SESSION_STARTED_AT=<provider-billing-start-epoch> $0 smoke-24k|smoke-32k|formal [output-directory]" >&2; exit 2 ;;
esac

: "${SESSION_STARTED_AT:?Set the actual provider billing-start Unix seconds}"
[[ "$SESSION_STARTED_AT" =~ ^[0-9]{10}$ ]] || { echo "Invalid billing start" >&2; exit 2; }
NOW=$(date +%s)
(( SESSION_STARTED_AT <= NOW )) || { echo "Billing start is in the future" >&2; exit 2; }
REMAINING=$((SESSION_STARTED_AT + 14400 - NOW))
(( REMAINING > 0 )) || { echo "Four-hour billing window expired" >&2; exit 2; }

PYTHON="$ROOT/.venv-qwen3/bin/python"
MANIFEST="$ROOT/manifests/route-v3-qwen3-model.json"
DATA="$ROOT/work/route-v3-qwen3/sft-short32k"
INVENTORY="$ROOT/work/qwen3-8b-model-inventory.json"
MODEL=/root/Qwen3-8B
OUTPUT=${2:-$ROOT/work/route-v3-qwen3/${MODE}-$(date -u +%Y%m%dT%H%M%SZ)}
[[ -x "$PYTHON" && -f "$MANIFEST" && -f "$DATA/manifest.json" && -f "$INVENTORY" ]] || {
  echo "Qwen3 environment, model manifest, data, or inventory is missing" >&2; exit 2;
}
[[ ! -e "$OUTPUT" ]] || { echo "Output already exists: $OUTPUT" >&2; exit 2; }

echo "mode=$MODE data=$DATA model=$MODEL output=$OUTPUT deadline=$((SESSION_STARTED_AT + 14400))"
exec timeout --signal=TERM --kill-after=30s "${REMAINING}s" \
  env TRAJECTORY_SFT_BOUNDED=1 "$PYTHON" -m coding_agent_rl_lab.trajectory_train \
  --data "$DATA" --output "$OUTPUT" --model-manifest "$MANIFEST" \
  --model-path "$MODEL" --inventory "$INVENTORY" "${TRAIN_ARGS[@]}"
