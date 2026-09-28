#!/usr/bin/env bash
set -euo pipefail

# Run the pinned SWE-Gym OpenHands fork on the Docker-capable Ubuntu VM.
# --preflight checks the CPU/runtime setup without contacting a model.
FORK_DIR="${SWEGYM_OPENHANDS_DIR:-/home/wesz/swegym-openhands}"
PIN="e644a2ca45c3623b27a7e6c169e3d479f0a87fbc"
TASK_ID="${SWEGYM_TASK_ID:-getmoto__moto-7646}"
IMAGE="xingyaoww/sweb.eval.x86_64.${TASK_ID//__/_s_}:latest"
export PATH="$HOME/swegym-bin:$HOME/.local/bin:$PATH"
export SWEGYM_USE_HOST_NETWORK=true

cd "$FORK_DIR"
test "$(git rev-parse HEAD)" = "$PIN" || { echo 'OpenHands commit mismatch' >&2; exit 1; }
python3 "$HOME/patch_swegym_host_network.py" "$FORK_DIR"
docker image inspect "$IMAGE" >/dev/null || { echo "Missing Docker image: $IMAGE" >&2; exit 1; }
poetry run python -c 'import openhands.agenthub, datasets, docker, litellm; from openhands.controller.agent import Agent; assert Agent.get_cls("CodeActAgent")'

if [[ "${1:-}" == '--preflight' ]]; then
  echo "Preflight passed: fork=$PIN task=$TASK_ID image=$IMAGE"
  exit 0
fi

: "${SESSION_STARTED_AT:?Set the actual provider billing-start Unix seconds}"
[[ "$SESSION_STARTED_AT" =~ ^[0-9]{10}$ ]] || { echo "Invalid billing start" >&2; exit 2; }
NOW=$(date +%s)
(( SESSION_STARTED_AT <= NOW )) || { echo "Billing start is in the future" >&2; exit 2; }
REMAINING=$((SESSION_STARTED_AT + 14400 - NOW))
(( REMAINING > 0 )) || { echo "Four-hour billing window expired" >&2; exit 2; }
: "${MODEL_ID:?Set MODEL_ID to the vLLM served model name}"
: "${MODEL_API_BASE:?Set MODEL_API_BASE to the OpenAI-compatible /v1 URL reachable from this VM}"

export MODEL_ID MODEL_API_BASE TASK_ID
export MODEL_TEMPERATURE=${MODEL_TEMPERATURE:-0.0}
MAX_ITERATIONS=${MAX_ITERATIONS:-50}
[[ "$MAX_ITERATIONS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid MAX_ITERATIONS" >&2; exit 2; }
poetry run python - <<'PY'
import os
from pathlib import Path
import toml

Path('config.toml').write_text(toml.dumps({
    'llm': {'swegym_local': {
        'model': 'openai/' + os.environ['MODEL_ID'],
        'api_key': 'EMPTY',
        'base_url': os.environ['MODEL_API_BASE'],
        'max_output_tokens': 2048,
        'temperature': float(os.environ['MODEL_TEMPERATURE']),
    }},
}), encoding='utf-8')
Path('evaluation/swe_bench/config.toml').write_text(
    toml.dumps({'selected_ids': [os.environ['TASK_ID']]}), encoding='utf-8'
)
PY
chmod 600 config.toml

export RUNTIME=eventstream
export USE_INSTANCE_IMAGE=true
export RUN_WITH_BROWSING=false
export USE_HINT_TEXT=false
export EVAL_DOCKER_IMAGE_PREFIX=docker.io/xingyaoww
export N_RUNS=1
export EXP_NAME=${EXP_NAME:-swegym-local-pilot}

REMAINING=$((SESSION_STARTED_AT + 14400 - $(date +%s)))
(( REMAINING > 0 )) || { echo "Four-hour billing window expired during setup" >&2; exit 2; }
timeout --signal=TERM --kill-after=30s "${REMAINING}s" ./evaluation/swe_bench/scripts/run_infer.sh \
  llm.swegym_local "$PIN" CodeActAgent 1 "$MAX_ITERATIONS" 1 SWE-Gym/SWE-Gym train 1
