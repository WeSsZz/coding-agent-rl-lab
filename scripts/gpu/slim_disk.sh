#!/usr/bin/env bash
# Optional package-cache cleanup only. Never deletes models, adapters or model caches.
# Dry run by default; --apply explicitly removes eligible package caches.
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
  'qwen'
  'huggingface'
  'modelscope'
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
  # Only rebuildable package caches. Models, HF/ModelScope caches and adapters are preserved.
  "$HOME_ROOT/.cache/pip"
  "$HOME_ROOT/.cache/uv"
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
