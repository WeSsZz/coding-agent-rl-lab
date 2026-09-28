#!/usr/bin/env bash
# Run a frozen SWE-Gym development task against the local SSH tunnel to AutoDL.
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$ROOT/envs/qwen3-serving.env"
MODE=${1:-}
case "$MODE" in
  base) MODEL=$QWEN3_BASE_MODEL ;;
  paired) MODEL=$QWEN3_PAIRED_MODEL ;;
  *) echo "Usage: SESSION_STARTED_AT=<provider-billing-start-epoch> $0 base|paired [output-directory]" >&2; exit 2 ;;
esac
: "${SESSION_STARTED_AT:?Set the actual provider billing-start Unix seconds}"
[[ "$SESSION_STARTED_AT" =~ ^[0-9]{10}$ ]] || { echo "Invalid billing start" >&2; exit 2; }
NOW=$(date +%s)
(( SESSION_STARTED_AT <= NOW )) || { echo "Billing start is in the future" >&2; exit 2; }
REMAINING=$((SESSION_STARTED_AT + 14400 - NOW))
(( REMAINING > 0 )) || { echo "Four-hour billing window expired" >&2; exit 2; }

TASK="$ROOT/work/route-v3-qwen3/getmoto__moto-7646.json"
PROMPT="$ROOT/envs/openhands-eval-template-v3.json"
OUTPUT=${2:-$ROOT/work/route-v3-qwen3/eval-${MODE}-$(date -u +%Y%m%dT%H%M%SZ)}
[[ -f "$TASK" && -f "$PROMPT" ]] || { echo "Frozen development task or prompt missing" >&2; exit 2; }
[[ ! -e "$OUTPUT" ]] || { echo "Output already exists: $OUTPUT" >&2; exit 2; }

ENDPOINT=${QWEN3_ENDPOINT:-http://127.0.0.1:18000}
"${PYTHON:-python3}" - "$ENDPOINT" "$MODEL" <<'PY'
import json, sys, urllib.request
endpoint, model = sys.argv[1:]
with urllib.request.urlopen(endpoint.rstrip('/') + '/v1/models', timeout=10) as response:
    models = json.load(response)
available = {item['id'] for item in models['data']}
if model not in available:
    raise SystemExit(f'{model} unavailable at {endpoint}; available={sorted(available)}')
PY

echo "mode=$MODE task=$TASK output=$OUTPUT deadline=$((SESSION_STARTED_AT + 14400))"
exec timeout --signal=TERM --kill-after=30s "${REMAINING}s" \
  "${PYTHON:-python3}" "$ROOT/scripts/evaluate_openhands_candidate.py" \
  --task "$TASK" --endpoint "$ENDPOINT" --model "$MODEL" --output "$OUTPUT" \
  --prompt-template "$PROMPT" --non-thinking --tool-choice auto --max-output-tokens 2048
