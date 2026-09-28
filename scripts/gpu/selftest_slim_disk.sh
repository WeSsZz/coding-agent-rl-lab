#!/usr/bin/env bash
# Prove that slim_disk.sh deletes what it should and protects what it must.
#
# This script destroys data, so its safety properties are tested the same way the rest of the
# project tests destructive code: build a fixture tree that mimics the real layout, run the
# script against it, and assert on what survived.
#
# Usage: bash scripts/gpu/selftest_slim_disk.sh
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SLIM="$SCRIPT_DIR/slim_disk.sh"

failures=0
check() {
  if [ "$2" = "1" ]; then
    echo "  ok    $1"
  else
    echo "  FAIL  $1"
    failures=$((failures + 1))
  fi
}

build_fixture() {
  local root=$1
  rm -rf "$root"
  mkdir -p "$root/autodl-tmp/qwen2.5-coder-14b-modelscope" "$root/autodl-tmp/sft60-lora"
  mkdir -p "$root/autodl-tmp/peft-0.21.0-overlay" "$root/.cache/pip" "$root/sftv24-evidence"
  # Make the disposable things clearly bigger than the protected ones so sizes are meaningful.
  dd if=/dev/zero of="$root/autodl-tmp/qwen2.5-coder-14b-modelscope/model.bin" bs=1024 count=200 status=none
  dd if=/dev/zero of="$root/.cache/pip/wheel.bin" bs=1024 count=100 status=none
  dd if=/dev/zero of="$root/autodl-tmp/sft60-lora/adapter_model.safetensors" bs=1024 count=40 status=none
  dd if=/dev/zero of="$root/autodl-tmp/peft-0.21.0-overlay/overlay.bin" bs=1024 count=5 status=none
  dd if=/dev/zero of="$root/sftv24-evidence/report.json" bs=1024 count=5 status=none
}

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "=== dry run must not delete anything ==="
build_fixture "$WORK"
HOME_ROOT="$WORK" AUTODL_DATA="$WORK/autodl-tmp" bash "$SLIM" >"$WORK/dry.log" 2>&1
check "model directory survives a dry run" "$([ -d "$WORK/autodl-tmp/qwen2.5-coder-14b-modelscope" ] && echo 1 || echo 0)"
check "pip cache survives a dry run" "$([ -d "$WORK/.cache/pip" ] && echo 1 || echo 0)"


echo "=== apply deletes the disposable bulk ==="
build_fixture "$WORK"
HOME_ROOT="$WORK" AUTODL_DATA="$WORK/autodl-tmp" bash "$SLIM" --apply >"$WORK/apply.log" 2>&1
check "model directory survives apply" "$([ -d "$WORK/autodl-tmp/qwen2.5-coder-14b-modelscope" ] && echo 1 || echo 0)"
check "pip cache is gone" "$([ ! -d "$WORK/.cache/pip" ] && echo 1 || echo 0)"

echo "=== apply must not touch the only copies ==="
check "LoRA adapter directory survives" "$([ -f "$WORK/autodl-tmp/sft60-lora/adapter_model.safetensors" ] && echo 1 || echo 0)"
check "peft overlay survives" "$([ -f "$WORK/autodl-tmp/peft-0.21.0-overlay/overlay.bin" ] && echo 1 || echo 0)"
check "evidence directory survives" "$([ -f "$WORK/sftv24-evidence/report.json" ] && echo 1 || echo 0)"

echo "=== a protected directory nested inside a candidate protects the candidate ==="
build_fixture "$WORK"
mkdir -p "$WORK/.cache/pip/keep-lora"
HOME_ROOT="$WORK" AUTODL_DATA="$WORK/autodl-tmp" bash "$SLIM" --apply >"$WORK/nested.log" 2>&1
check "candidate with a protected child survives" "$([ -d "$WORK/.cache/pip/keep-lora" ] && echo 1 || echo 0)"
check "the skip is reported" "$(grep -q 'contains protected path' "$WORK/nested.log" && echo 1 || echo 0)"

echo "=== a missing candidate is not an error ==="
build_fixture "$WORK"
HOME_ROOT="$WORK" AUTODL_DATA="$WORK/autodl-tmp" bash "$SLIM" --apply >"$WORK/missing.log" 2>&1
check "run completes with absent candidates" "$(grep -q 'reclaimed summary' "$WORK/missing.log" && echo 1 || echo 0)"

echo
if [ $failures -eq 0 ]; then
  echo "RESULT: OK"
  exit 0
fi
echo "RESULT: FAILED ($failures)"
exit 1
