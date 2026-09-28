#!/usr/bin/env bash
# Locate model weights that are already on the box.
#
# The cloud instance already carries Qwen2.5 checkpoints from route v1, so the default plan is to
# USE them, not to download anything. Downloading was only ever a fallback, and establishing that
# is the difference between a two-minute check and four hours of transfer.
#
# Reports, per candidate directory: size, whether it has the files a loader needs, and how many
# safetensors shards it holds. A directory can look plausible and still be a partial download, so
# "complete" is decided by the presence of config + tokenizer + at least one shard.
#
# Usage: bash scripts/gpu/find_local_model.sh [search-root ...]
set -uo pipefail

ROOTS=("$@")
if [ ${#ROOTS[@]} -eq 0 ]; then
  ROOTS=(/root /root/autodl-tmp /root/autodl-fs /home /mnt /data)
fi

echo "=== searching for model directories ==="
echo "roots: ${ROOTS[*]}"
echo

FOUND=0
for root in "${ROOTS[@]}"; do
  [ -d "$root" ] || continue
  while IFS= read -r dir; do
    [ -n "$dir" ] || continue
    FOUND=$((FOUND + 1))

    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    shards=$(find "$dir" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l)
    has_config=$([ -f "$dir/config.json" ] && echo yes || echo no)
    has_tokenizer=$([ -f "$dir/tokenizer.json" ] || [ -f "$dir/tokenizer_config.json" ] && echo yes || echo no)
    index=$([ -f "$dir/model.safetensors.index.json" ] && echo yes || echo no)

    status="INCOMPLETE"
    if [ "$has_config" = "yes" ] && [ "$has_tokenizer" = "yes" ] && [ "$shards" -ge 1 ]; then
      status="LOADABLE"
    fi

    printf '%-10s %-6s shards=%-3s config=%-3s tokenizer=%-3s index=%-3s %s\n' \
      "$status" "$size" "$shards" "$has_config" "$has_tokenizer" "$index" "$dir"

    # Print the model type stated by the config, which is what decides 7B vs 14B vs something else.
    if [ "$has_config" = "yes" ] && command -v python3 >/dev/null 2>&1; then
      python3 - "$dir/config.json" <<'PY' 2>/dev/null || true
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(0)
print(f"           model_type={data.get('model_type')} "
      f"hidden={data.get('hidden_size')} layers={data.get('num_hidden_layers')} "
      f"dtype={data.get('torch_dtype') or data.get('dtype')}")
PY
    fi
  done < <(find "$root" -maxdepth 4 -type d \
             \( -iname '*qwen*coder*' -o -iname '*qwen2.5*' -o -iname '*Qwen3*' \) \
             -not -path '*/.git/*' 2>/dev/null)
done

echo
if [ "$FOUND" -eq 0 ]; then
  echo "no model directory found under: ${ROOTS[*]}"
  echo "widen the search, or fall back to scripts/gpu/fetch_model.sh"
  exit 1
fi

echo "=== how to use the result ==="
echo "  export MODEL_PATH=<a LOADABLE path above>"
echo "  a LOADABLE directory needs no download; do not fetch a second copy"
