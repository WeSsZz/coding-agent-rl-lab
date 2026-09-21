#!/usr/bin/env bash
# Reclaim disk on a cloned GPU instance, safely.
#
# Route v1 left several tens of gigabytes of Qwen2.5-Coder weights that route v2 will never
# load, plus caches and logs. This script deletes only from an explicit allow-list, protects an
# explicit keep-list, prints what it would do by default, and requires `--apply` to act.
#
# Why an allow-list rather than "find and rm": the clone also carries the only copies of the
# trained LoRA adapters and the environment record that makes route v1 re-runnable at all.
# A pattern-based delete is exactly how those get destroyed by accident.
#
# Usage:
#   bash scripts/gpu/slim_disk.sh                 # dry run: print what would go
#   bash scripts/gpu/slim_disk.sh --apply         # actually delete
#   bash scripts/gpu/slim_disk.sh --apply --keep-instance-record   # default; see KEEP below
set -uo pipefail

APPLY=0
for argument in "$@"; do
  case "$argument" in
    --apply) APPLY=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $argument" >&2; exit 2 ;;
  esac
done

AUTODL=${AUTODL_DATA:-/root/autodl-tmp}
HOME_ROOT=${HOME_ROOT:-/root}

# Anything matching one of these is never deleted, no matter what else says otherwise.
KEEP_PATTERNS=(
  'lora'
  'adapter'
  'peft-0.21.0-overlay'
  'evidence'
  'report'
  'audit'
  'token'
  'coding-agent-rl-lab'
  'snapshot'
)

# Explicit, re-downloadable bulk. Each entry is removed only if it exists.
CANDIDATES=(
  # Old route's policy models. Route v2 uses Qwen/Qwen3-8B; these are re-downloadable.
  "$AUTODL/qwen2.5-coder-7b-modelscope"
  "$AUTODL/qwen2.5-coder-14b-modelscope"
  "$AUTODL/qwen2.5-coder-7b"
  "$AUTODL/qwen2.5-coder-14b"
  # Caches: regenerated on demand.
  "$HOME_ROOT/.cache/pip"
  "$HOME_ROOT/.cache/uv"
  "$HOME_ROOT/.cache/huggingface/hub"
  "$HOME_ROOT/.cache/modelscope"
)

is_protected() {
  local target=$1
  local lowered
  lowered=$(printf '%s' "$target" | tr '[:upper:]' '[:lower:]')
  for pattern in "${KEEP_PATTERNS[@]}"; do
    case "$lowered" in
      *"$pattern"*) return 0 ;;
    esac
  done
  return 1
}

echo "mode: $([ $APPLY -eq 1 ] && echo APPLY || echo 'DRY RUN')"
echo
echo "=== protected patterns ==="
printf '  %s\n' "${KEEP_PATTERNS[@]}"
echo

freed_kib=0
deleted=0
skipped=0

for target in "${CANDIDATES[@]}"; do
  if [ ! -e "$target" ]; then
    continue
  fi
  size_kib=$(du -sk "$target" 2>/dev/null | cut -f1)
  size_h=$(du -sh "$target" 2>/dev/null | cut -f1)

  if is_protected "$target"; then
    echo "PROTECTED  $size_h  $target"
    skipped=$((skipped + 1))
    continue
  fi

  # Never delete a directory that still holds a protected subdirectory.
  protected_inside=$(find "$target" -maxdepth 4 2>/dev/null | while read -r path; do
    is_protected "$path" && { echo "$path"; break; }
  done)
  if [ -n "$protected_inside" ]; then
    echo "SKIP       $size_h  $target  (contains protected path: $protected_inside)"
    skipped=$((skipped + 1))
    continue
  fi

  if [ $APPLY -eq 1 ]; then
    if rm -rf -- "$target"; then
      echo "deleted    $size_h  $target"
      freed_kib=$((freed_kib + size_kib))
      deleted=$((deleted + 1))
    else
      echo "FAILED     $size_h  $target"
    fi
  else
    echo "would free $size_h  $target"
    freed_kib=$((freed_kib + size_kib))
    deleted=$((deleted + 1))
  fi
done

echo
echo "=== also worth checking by hand ==="
echo "  - docker images and the rayon/rust build caches:  docker system df"
echo "  - stray rollout outputs already archived in remote-artifacts/"
echo

echo "=== reclaimed summary ==="
printf '  entries %d, protected/skipped %d, total %s\n' "$deleted" "$skipped" "$(awk -v k="$freed_kib" 'BEGIN{printf "%.1f GiB", k/1024/1024}')"

if [ $APPLY -eq 0 ]; then
  echo
  echo "this was a dry run; re-run with --apply to delete"
fi

echo
echo "=== disk after ==="
df -h / 2>/dev/null | tail -1
[ -d "$AUTODL" ] && df -h "$AUTODL" 2>/dev/null | tail -1
