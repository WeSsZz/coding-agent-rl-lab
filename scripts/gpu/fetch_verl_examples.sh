#!/usr/bin/env bash
# Fetch the veRL example scripts this project intends to run, so the plan quotes real files.
set -uo pipefail
REV=1876b06d0a3e4e71e06230be10af14492ca8a75b
BASE=https://raw.githubusercontent.com/verl-project/verl/$REV
OUT=${1:-/tmp/verl}
mkdir -p "$OUT"

for path in \
  examples/grpo_trainer/run_qwen3_8b_fsdp.sh \
  examples/tuning/lora/run_qwen3_8b_fsdp.sh \
  examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh \
  verl/trainer/main_ppo.py ; do
  name=$(echo "$path" | tr '/' '_')
  printf '%-58s ' "$path"
  code=$(curl -sL --max-time 40 -o "$OUT/$name" -w '%{http_code}' "$BASE/$path")
  printf '%s %s bytes\n' "$code" "$(wc -c < "$OUT/$name" 2>/dev/null || echo 0)"
done
