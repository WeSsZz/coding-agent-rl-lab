#!/usr/bin/env bash
# Prepare pinned dependencies/model. All runtime counts toward the paid deadline.
# No automatic deletion. MODEL_DIR may select an existing snapshot; it is fully verified.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
WORK=${GPU_WORK:-/root/autodl-tmp/route-v2}
REPORT="$WORK/setup-report.json"
LOG="$WORK/setup.log"

VERL_COMMIT=1876b06d0a3e4e71e06230be10af14492ca8a75b
VERL_DIR="$WORK/verl"
PINNED_MODEL=$(python3 "$REPO_ROOT/scripts/model_contract.py" --field id)
PINNED_REVISION=$(python3 "$REPO_ROOT/scripts/model_contract.py" --field revision)
MODEL_ID=${MODEL_ID:-$PINNED_MODEL}
[ "$MODEL_ID" = "$PINNED_MODEL" ] || { echo "MODEL_ID differs from manifest" >&2; exit 2; }
MODEL_DIR=${MODEL_DIR:-/root/autodl-tmp/models/${MODEL_ID##*/}}

APPLY_SLIM=0
SKIP_MODEL=0
for argument in "$@"; do
  case "$argument" in
    --slim) echo "Automatic deletion disabled; preserve existing model weights." >&2; exit 2 ;;
    --skip-model) SKIP_MODEL=1 ;;
    -h|--help) echo 'Usage: SESSION_STARTED_AT=<billing epoch> MODEL_DIR=<snapshot> bash setup_instance.sh [--skip-model]'; exit 0 ;;
    *) echo "unknown argument: $argument" >&2; exit 2 ;;
  esac
done
# Epoch recorded at provider billing start, not after installation.
: "${SESSION_STARTED_AT:?Set SESSION_STARTED_AT to provider billing-start Unix seconds}"
[[ "$SESSION_STARTED_AT" =~ ^[0-9]{10}$ ]] || { echo "Invalid billing-start epoch" >&2; exit 2; }
[ "$SESSION_STARTED_AT" -le "$(date +%s)" ] || { echo "Billing start cannot be in the future" >&2; exit 2; }
DEADLINE=$((SESSION_STARTED_AT + 14400))
[ "$(date +%s)" -lt "$DEADLINE" ] || { echo "Session deadline already reached" >&2; exit 2; }
if [ "${SETUP_BOUNDED:-0}" != 1 ]; then
  exec timeout --signal=TERM --kill-after=30s "$((DEADLINE - $(date +%s)))s" env SETUP_BOUNDED=1 bash "$0" "$@"
fi

mkdir -p "$WORK"
exec > >(tee -a "$LOG") 2>&1

echo "########################################################################"
echo "# route v2 instance bring-up   $(date -u +%FT%TZ)"
echo "# work=$WORK  slim=$APPLY_SLIM  skip_model=$SKIP_MODEL"
echo "########################################################################"

STEPS_OK=()
STEPS_FAILED=()
trap 'code=$?; printf "FAILED exit=%s line=%s\n" "$code" "$LINENO" > "$WORK/setup-failure.txt"; exit "$code"' ERR
trap 'printf "Setup terminated at session deadline or external signal\n" > "$WORK/setup-failure.txt"; exit 124' TERM

record() {
  # record <name> <ok|failed>
  if [ "$2" = "ok" ]; then STEPS_OK+=("$1"); else STEPS_FAILED+=("$1"); fi
}

# ---------------------------------------------------------------- 1. preflight
echo
echo "=== [1/5] preflight ==="
if command -v python3 >/dev/null 2>&1; then
  python3 "$REPO_ROOT/scripts/preflight_gpu_env.py" --json "$WORK/preflight.json"
  record preflight ok
else
  echo "python3 missing"
  record preflight failed
  exit 1
fi

# ---------------------------------------------------------------- 2. inventory
echo
echo "=== [2/5] disk inventory ==="
bash "$REPO_ROOT/scripts/gpu/inventory_disk.sh" | tee "$WORK/inventory.txt"
record inventory ok

# No automatic cleanup: inventory is read-only. Reuse verified weights first.
echo "Automatic disk cleanup disabled; existing weights and caches are preserved."

# ---------------------------------------------------------------- 4. verl
echo
echo "=== [4/5] veRL at $VERL_COMMIT ==="
if [ ! -d "$VERL_DIR/.git" ]; then
  git clone https://github.com/verl-project/verl.git "$VERL_DIR"
fi
if [ -d "$VERL_DIR/.git" ]; then
  git -C "$VERL_DIR" fetch --all --tags --quiet
  git -C "$VERL_DIR" checkout --quiet "$VERL_COMMIT"
  actual=$(git -C "$VERL_DIR" rev-parse HEAD)
  echo "  checked out $actual"
  if [ "$actual" = "$VERL_COMMIT" ]; then record "verl-checkout" ok; else record "verl-checkout" failed; exit 1; fi

  echo "  uv.lock present: $([ -f "$VERL_DIR/uv.lock" ] && echo yes || echo NO)"
  if ! command -v uv >/dev/null 2>&1; then
    echo "  installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh || python3 -m pip install --quiet uv
    export PATH="$HOME/.local/bin:$PATH"
  fi
  echo "  uv: $(command -v uv || echo missing)"

  echo "  materialising the environment (this downloads torch/vllm/flash-attn; expect 10-30 min)"
  START=$(date +%s)
  ( cd "$VERL_DIR" && uv run --frozen --all-packages --extra vllm --extra fsdp python3 -c "
import importlib.metadata as m
for name in ('torch','vllm','transformers','ray','peft','tensordict','verl'):
    try:
        print(f'  {name:14s} {m.version(name)}')
    except Exception:
        raise
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
print('  torch.cuda.is_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  device:', torch.cuda.get_device_name(0))
    print('  total GiB:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1))
" )
  STATUS=$?
  END=$(date +%s)
  echo "  environment step took $((END - START))s, exit $STATUS"
  if [ $STATUS -eq 0 ]; then record verl-env ok; else record verl-env failed; fi
else
  echo "  clone failed"
  record "verl-checkout" failed
fi

# ---------------------------------------------------------------- 5. model
echo
echo "=== [5/5] model weights ==="
echo "looking for weights that are already on this box before considering a download"
echo
# Discovery is informational; never select the first plausible directory automatically.
if ! bash "$SCRIPT_DIR/find_local_model.sh" | tee "$WORK/local-models.txt"; then
  echo "No local candidate found; only the explicit MODEL_DIR is used."
fi
# A caller can point MODEL_DIR at an existing snapshot. Wrong/incomplete snapshots fail closed.
bash "$SCRIPT_DIR/verify_model_revision.sh" "$MODEL_ID" "$PINNED_REVISION"
INVENTORY="$REPO_ROOT/work/model-inventory.json"
if [ ! -f "$MODEL_DIR/config.json" ]; then
  if [ $SKIP_MODEL -eq 1 ]; then
    echo "Model absent and downloading disabled: $MODEL_DIR" >&2
    exit 1
  fi
  MODEL_ID="$MODEL_ID" REVISION="$PINNED_REVISION" bash "$SCRIPT_DIR/fetch_model.sh" "$MODEL_DIR"
fi
python3 "$REPO_ROOT/scripts/model_contract.py" --directory "$MODEL_DIR" \
  --inventory "$INVENTORY" --output "$WORK/model-verification.json"
printf '%s\n' "$MODEL_DIR" > "$WORK/verified-model-path.txt"
record model ok

# ---------------------------------------------------------------- report
echo
echo "########################################################################"
echo "# summary   $(date -u +%FT%TZ)"
echo "# ok     : ${STEPS_OK[*]:-none}"
echo "# failed : ${STEPS_FAILED[*]:-none}"
echo "########################################################################"

python3 - "$REPORT" "${STEPS_OK[*]:-}" "${STEPS_FAILED[*]:-}" <<'PY' 2>/dev/null
import json, sys, time, pathlib, subprocess

report_path, ok, failed = sys.argv[1], sys.argv[2], sys.argv[3]

def command(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None

report = {
    "schema_version": 1,
    "manifest_id": "route-v2-gpu-instance-setup",
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "steps_ok": ok.split() if ok else [],
    "steps_failed": failed.split() if failed else [],
    "host": command(["hostname"]),
    "gpu": command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader"]),
    "disk": command(["df", "-h", "/"]),
}
pathlib.Path(report_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"wrote {report_path}")
PY

echo
df -h / /root/autodl-tmp 2>/dev/null | sort -u

if [ ${#STEPS_FAILED[@]} -eq 0 ]; then
  echo "RESULT: OK - the instance is dependencies/model verified; SFT data and agent integration are separate gates"
  exit 0
fi
echo "RESULT: FAILED - fix: ${STEPS_FAILED[*]}"
echo "the log is at $LOG"
exit 1
