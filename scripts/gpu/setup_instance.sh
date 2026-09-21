#!/usr/bin/env bash
# Bring a freshly cloned AutoDL instance to the point where a timed training session can start.
#
# Everything here is untimed bring-up: it is designed to be run once, re-run safely after a
# failure, and log enough that the timed session does not have to rediscover anything.
#
# Order is deliberate:
#   1. preflight   - is the driver/disk/model-source situation what the plan assumes?
#   2. inventory   - what did the clone actually bring?
#   3. slim        - free the disk BEFORE fetching a 16 GB model and a ~25 GB venv
#   4. verl        - clone at the pinned commit and materialise the uv environment
#   5. model       - fetch Qwen3-8B and verify it against the pinned config hashes
#
# Slimming runs in dry-run mode unless --slim is passed, because it deletes weights and this
# project has already been burned once by an environment it could not rebuild.
#
# Usage (as root on the GPU box):
#   bash setup_instance.sh                 # preflight + inventory + dry-run slim + verl + model
#   bash setup_instance.sh --slim          # actually delete the disposable weights
#   bash setup_instance.sh --skip-model    # stop before the 16 GB download
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
WORK=${GPU_WORK:-/root/autodl-tmp/route-v2}
REPORT="$WORK/setup-report.json"
LOG="$WORK/setup.log"

VERL_COMMIT=1876b06d0a3e4e71e06230be10af14492ca8a75b
VERL_DIR="$WORK/verl"
MODEL_DIR=${MODEL_DIR:-/root/autodl-tmp/models/Qwen3-8B}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-8B}

APPLY_SLIM=0
SKIP_MODEL=0
for argument in "$@"; do
  case "$argument" in
    --slim) APPLY_SLIM=1 ;;
    --skip-model) SKIP_MODEL=1 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown argument: $argument" >&2; exit 2 ;;
  esac
done

mkdir -p "$WORK"
exec > >(tee -a "$LOG") 2>&1

echo "########################################################################"
echo "# route v2 instance bring-up   $(date -u +%FT%TZ)"
echo "# work=$WORK  slim=$APPLY_SLIM  skip_model=$SKIP_MODEL"
echo "########################################################################"

STEPS_OK=()
STEPS_FAILED=()

record() {
  # record <name> <ok|failed>
  if [ "$2" = "ok" ]; then STEPS_OK+=("$1"); else STEPS_FAILED+=("$1"); fi
}

# ---------------------------------------------------------------- 1. preflight
echo
echo "=== [1/5] preflight ==="
if command -v python3 >/dev/null 2>&1; then
  python3 "$REPO_ROOT/scripts/preflight_gpu_env.py" --json "$WORK/preflight.json"
  if [ $? -eq 0 ]; then record preflight ok; else record preflight failed; fi
else
  echo "python3 missing"
  record preflight failed
fi

# ---------------------------------------------------------------- 2. inventory
echo
echo "=== [2/5] disk inventory ==="
bash "$REPO_ROOT/scripts/gpu/inventory_disk.sh" | tee "$WORK/inventory.txt"
record inventory ok

# ---------------------------------------------------------------- 3. slim
echo
echo "=== [3/5] disk slimming ==="
if [ $APPLY_SLIM -eq 1 ]; then
  bash "$SCRIPT_DIR/slim_disk.sh" --apply
else
  echo "(dry run; re-run this script with --slim to actually delete)"
  bash "$SCRIPT_DIR/slim_disk.sh"
fi
record slim ok

# ---------------------------------------------------------------- 4. verl
echo
echo "=== [4/5] veRL at $VERL_COMMIT ==="
if [ ! -d "$VERL_DIR/.git" ]; then
  git clone https://github.com/verl-project/verl.git "$VERL_DIR" || true
fi
if [ -d "$VERL_DIR/.git" ]; then
  git -C "$VERL_DIR" fetch --all --tags --quiet || true
  git -C "$VERL_DIR" checkout --quiet "$VERL_COMMIT"
  actual=$(git -C "$VERL_DIR" rev-parse HEAD)
  echo "  checked out $actual"
  if [ "$actual" = "$VERL_COMMIT" ]; then record "verl-checkout" ok; else record "verl-checkout" failed; fi

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
    except Exception as exc:
        print(f'  {name:14s} MISSING ({exc})')
import torch
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
echo "=== [5/5] Qwen3-8B ==="
if [ $SKIP_MODEL -eq 1 ]; then
  echo "(skipped)"
elif [ -f "$MODEL_DIR/config.json" ]; then
  echo "  already present at $MODEL_DIR"
  bash "$SCRIPT_DIR/fetch_model.sh" "$MODEL_DIR" >/dev/null 2>&1 && record model ok || record model failed
else
  MODEL_ID="$MODEL_ID" bash "$SCRIPT_DIR/fetch_model.sh" "$MODEL_DIR"
  if [ $? -eq 0 ]; then record model ok; else record model failed; fi
fi

# ---------------------------------------------------------------- report
echo
echo "########################################################################"
echo "# summary   $(date -u +%FT%TZ)"
echo "# ok     : ${STEPS_OK[*]:-none}"
echo "# failed : ${STEPS_FAILED[*]:-none}"
echo "########################################################################"

python3 - "$REPORT" "${STEPS_OK[*]:-}" "${STEPS_FAILED[*]:-}" <<'PY' 2>/dev/null || true
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
  echo "RESULT: OK - the instance is ready for the timed session"
  exit 0
fi
echo "RESULT: FAILED - fix: ${STEPS_FAILED[*]}"
echo "the log is at $LOG"
exit 1
