#!/usr/bin/env bash
# Measure where a model can actually be fetched from fastest.
#
# A 15 GB download is not a formality: at the 1 MB/s that hf-mirror delivered to the worker, it
# is over four hours. If the GPU box shares that, downloading inside a metered session is the
# single most expensive thing in the plan. This measures the candidates so the choice is made on
# numbers.
#
# Usage: bash scripts/gpu/measure_model_sources.sh [model-id]
set -uo pipefail

MODEL=${1:-Qwen/Qwen2.5-Coder-7B-Instruct}
#: A file that exists in every Qwen repo and is small enough to fetch repeatedly.
PROBE_FILE=${PROBE_FILE:-config.json}
SECONDS_TO_MEASURE=${SECONDS_TO_MEASURE:-15}

measure() {
  local name=$1 url=$2
  printf '%-46s ' "$name"
  local out
  out=$(curl -sL --max-time "$SECONDS_TO_MEASURE" -o /dev/null \
        -w '%{http_code} %{size_download} %{speed_download}' "$url" 2>/dev/null) || out="000 0 0"
  local code size speed
  code=$(echo "$out" | awk '{print $1}')
  size=$(echo "$out" | awk '{print $2}')
  speed=$(echo "$out" | awk '{print $3}')
  if [ "$code" = "200" ]; then
    printf 'ok    %8s bytes  %6.0f KB/s\n' "$size" "$(echo "$speed" | awk '{print $1/1024}')"
  else
    printf 'http %s\n' "$code"
  fi
}

echo "model: $MODEL"
echo "probe: $PROBE_FILE, up to ${SECONDS_TO_MEASURE}s per host"
echo

ORG=${MODEL%%/*}
NAME=${MODEL##*/}

measure "huggingface.co (direct)" \
  "https://huggingface.co/$MODEL/resolve/main/$PROBE_FILE"
measure "hf-mirror.com" \
  "https://hf-mirror.com/$MODEL/resolve/main/$PROBE_FILE"
measure "modelscope.cn api v1" \
  "https://www.modelscope.cn/api/v1/models/$MODEL/repo?Revision=master&FilePath=$PROBE_FILE"
measure "modelscope.cn models path" \
  "https://modelscope.cn/models/$MODEL/resolve/master/$PROBE_FILE"

echo
echo "=== a bigger file, to get a real throughput figure rather than a latency figure ==="
BIG=models-00001-of-00004.safetensors
for label in "hf-mirror" "modelscope"; do
  case "$label" in
    hf-mirror) url="https://hf-mirror.com/$MODEL/resolve/main/$BIG" ;;
    modelscope) url="https://www.modelscope.cn/api/v1/models/$MODEL/repo?Revision=master&FilePath=$BIG" ;;
  esac
  printf '%-46s ' "$label ($BIG)"
  out=$(curl -sL --max-time "$SECONDS_TO_MEASURE" -o /dev/null -w '%{http_code} %{size_download} %{speed_download}' "$url" 2>/dev/null) || out="000 0 0"
  code=$(echo "$out" | awk '{print $1}')
  size=$(echo "$out" | awk '{print $2}')
  speed=$(echo "$out" | awk '{print $3}')
  if [ "$code" = "200" ]; then
    printf 'ok    %10s bytes  %6.0f KB/s\n' "$size" "$(echo "$speed" | awk '{print $1/1024}')"
  else
    printf 'http %s\n' "$code"
  fi
done

echo
echo "interpretation: MB/s = KB/s / 1024. At 1 MB/s a 15 GB model takes about 4.3 hours."
