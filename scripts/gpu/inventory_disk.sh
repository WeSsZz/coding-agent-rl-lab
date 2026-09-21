#!/usr/bin/env bash
# Read-only inventory of a GPU box: what is eating the disk, and what must not be touched.
#
# The old AutoDL instance is being cloned onto a new GPU, so its data disk arrives with several
# tens of gigabytes of model weights that route v2 will never load (route v1 used Qwen2.5-Coder;
# route v2 uses Qwen3-8B). This script only looks. Deletion is a separate, dry-run-first script.
#
# Usage: bash scripts/gpu/inventory_disk.sh [root]
set -uo pipefail

ROOT=${1:-/root}
AUTODL=${AUTODL_DATA:-/root/autodl-tmp}

echo "=== filesystems ==="
df -h / "$AUTODL" 2>/dev/null | sort -u

echo
echo "=== top-level usage under $ROOT (depth 1) ==="
du -xhd1 "$ROOT" 2>/dev/null | sort -rh | head -25

if [ -d "$AUTODL" ]; then
  echo
  echo "=== top-level usage under $AUTODL (depth 1) ==="
  du -xhd1 "$AUTODL" 2>/dev/null | sort -rh | head -30
fi

echo
echo "=== candidate model directories (large, re-downloadable) ==="
for base in "$ROOT" "$AUTODL"; do
  [ -d "$base" ] || continue
  find "$base" -maxdepth 3 -type d \
    \( -name '*modelscope*' -o -name '*Qwen*' -o -name '*qwen*' -o -name 'models--*' -o -name 'hub' \) \
    -prune 2>/dev/null | while read -r dir; do
      printf '%10s  %s\n' "$(du -sh "$dir" 2>/dev/null | cut -f1)" "$dir"
    done
done

echo
echo "=== protected: training artefacts that exist nowhere else ==="
for base in "$ROOT" "$AUTODL"; do
  [ -d "$base" ] || continue
  find "$base" -maxdepth 4 -type d \
    \( -iname '*lora*' -o -iname '*adapter*' -o -iname '*peft*' -o -iname '*evidence*' \) \
    2>/dev/null | while read -r dir; do
      printf '%10s  %s\n' "$(du -sh "$dir" 2>/dev/null | cut -f1)" "$dir"
    done
done

echo
echo "=== python environments ==="
find "$ROOT" -maxdepth 3 -name 'pyvenv.cfg' 2>/dev/null | while read -r cfg; do
  env_dir=$(dirname "$cfg")
  printf '%10s  %s  (%s)\n' "$(du -sh "$env_dir" 2>/dev/null | cut -f1)" "$env_dir" "$(grep -m1 '^version' "$cfg" | cut -d' ' -f3)"
done

echo
echo "=== caches (safe to drop, regenerated on demand) ==="
for cache in "$ROOT/.cache/pip" "$ROOT/.cache/huggingface" "$ROOT/.cache/modelscope" "$ROOT/.cache/uv"; do
  [ -e "$cache" ] && printf '%10s  %s\n' "$(du -sh "$cache" 2>/dev/null | cut -f1)" "$cache"
done

echo
echo "=== training reports and logs (small; keep) ==="
find "$ROOT" -maxdepth 3 -type f \( -name '*.log' -o -name '*report*.json' -o -name '*audit*.jsonl' \) -size +1k 2>/dev/null \
  | head -20 | while read -r file; do
    printf '%10s  %s\n' "$(du -h "$file" 2>/dev/null | cut -f1)" "$file"
  done

echo
echo "=== docker ==="
if command -v docker >/dev/null 2>&1; then
  docker system df 2>/dev/null || echo "(docker present but not usable by this user)"
else
  echo "(no docker on this box)"
fi

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "(nvidia-smi unavailable)"
nvidia-smi 2>/dev/null | grep -i 'CUDA Version' || true
