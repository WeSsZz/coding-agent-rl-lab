#!/usr/bin/env bash
# GPU-only vLLM endpoint for matched Qwen3 base / LoRA evaluation.
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$ROOT/envs/qwen3-serving.env"
MODE=${1:-}
case "$MODE" in base|paired) ;; *) echo "Usage: SESSION_STARTED_AT=<provider-billing-start-epoch> $0 base|paired [adapter-directory]" >&2; exit 2 ;; esac
: "${SESSION_STARTED_AT:?Set the actual provider billing-start Unix seconds}"
[[ "$SESSION_STARTED_AT" =~ ^[0-9]{10}$ ]] || { echo "Invalid billing start" >&2; exit 2; }
NOW=$(date +%s)
(( SESSION_STARTED_AT <= NOW )) || { echo "Billing start is in the future" >&2; exit 2; }
REMAINING=$((SESSION_STARTED_AT + 14400 - NOW))
(( REMAINING > 0 )) || { echo "Four-hour billing window expired" >&2; exit 2; }

MODEL=/root/Qwen3-8B
ROLLOUT_TEMPLATE="$ROOT/envs/qwen3-8b-nonthinking-rollout.jinja"
VLLM="$ROOT/.venv-qwen3/bin/vllm"
[[ -x "$VLLM" ]] || { echo "vLLM is not installed in the Qwen3 environment" >&2; exit 2; }
[[ -f "$ROLLOUT_TEMPLATE" ]] || { echo "Qwen3 non-thinking rollout template missing" >&2; exit 2; }
printf '%s  %s\n' '0389654058fcc7dddb8b2faaf20ac27a3cd81f6abfe014a116fff4ef2e8e1214' "$ROLLOUT_TEMPLATE" | sha256sum -c - >/dev/null
[[ -f "$ROOT/work/qwen3-8b-system-verification.json" ]] || { echo "Verified Qwen3 snapshot report missing" >&2; exit 2; }
"$ROOT/.venv-qwen3/bin/python" "$ROOT/scripts/model_contract.py" \
  --manifest "$ROOT/manifests/route-v3-qwen3-model.json" --directory "$MODEL" \
  --inventory "$ROOT/work/qwen3-8b-model-inventory.json" >/dev/null
REMAINING=$((SESSION_STARTED_AT + 14400 - $(date +%s)))
(( REMAINING > 0 )) || { echo "Four-hour billing window expired during model verification" >&2; exit 2; }

EXTRA=()
if [[ "$MODE" == paired ]]; then
  ADAPTER=${2:?Provide the formal Qwen3 adapter directory}
  [[ -f "$ADAPTER/adapter_model.safetensors" && -f "$ADAPTER/adapter_config.json" ]] || {
    echo "Qwen3 adapter files missing: $ADAPTER" >&2; exit 2;
  }
  EXTRA=(--enable-lora --max-lora-rank 16 --lora-modules "$QWEN3_PAIRED_MODEL=$ADAPTER")
fi

echo "Serving Qwen3 $MODE on ${BIND_HOST:-127.0.0.1}:${PORT:-8000}; deadline=$((SESSION_STARTED_AT + 14400))"
exec timeout --signal=TERM --kill-after=30s "${REMAINING}s" "$VLLM" serve "$MODEL" \
  --served-model-name "$QWEN3_BASE_MODEL" --host "${BIND_HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --chat-template "$ROLLOUT_TEMPLATE" \
  --enable-auto-tool-choice --tool-call-parser hermes "${EXTRA[@]}"
