#!/usr/bin/env bash
set -Eeuo pipefail

FORK_DIR="${SWEGYM_OPENHANDS_DIR:-/home/wesz/swegym-openhands}"
PIN="e644a2ca45c3623b27a7e6c169e3d479f0a87fbc"
INPUT=${1:?Usage: $0 <prediction.jsonl>}
INPUT=$(realpath "$INPUT")
[[ -f "$INPUT" ]] || { echo "Prediction file missing: $INPUT" >&2; exit 2; }
export PATH="$HOME/swegym-bin:$HOME/.local/bin:$PATH"
export SWEGYM_USE_HOST_NETWORK=true
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd "$FORK_DIR"
test "$(git rev-parse HEAD)" = "$PIN" || { echo 'OpenHands commit mismatch' >&2; exit 1; }
python3 "$HOME/patch_swegym_host_network.py" "$FORK_DIR"

exec timeout --signal=TERM --kill-after=30s 3600s poetry run python \
  evaluation/swe_bench/eval_infer.py \
  --input-file "$INPUT" --dataset SWE-Gym/SWE-Gym --split train \
  --eval-num-workers 1 --eval-n-limit 1
