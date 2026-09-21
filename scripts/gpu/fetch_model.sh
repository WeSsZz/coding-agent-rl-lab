#!/usr/bin/env bash
# Fetch Qwen3-8B weights on the GPU box, preferring whichever host actually answers.
#
# The model is NOT pre-installed. The old instance carries Qwen2.5-Coder snapshots from route
# v1; route v2 uses Qwen3-8B, about 16 GB of bfloat16 safetensors that has to be downloaded.
# `huggingface.co` is unreachable from this project's networks, so the mirror path is the
# expected one.
#
# The download is verified against the two files this repository already pinned:
#   config.json            sha256 f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30
#   tokenizer_config.json  sha256 d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101
# A mismatch means the mirror served a different revision than the one the chat template was
# pinned from, which would silently change every prompt.
#
# Usage:
#   bash scripts/gpu/fetch_model.sh [target-dir]
set -uo pipefail

MODEL=${MODEL_ID:-Qwen/Qwen3-8B}
TARGET=${1:-/root/autodl-tmp/models/Qwen3-8B}
EXPECT_CONFIG=f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30
EXPECT_TOKENIZER=d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101

echo "=== target ==="
echo "  $TARGET"
mkdir -p "$TARGET"

echo
echo "=== choosing an endpoint ==="
ENDPOINT=""
for candidate in "https://hf-mirror.com" "https://huggingface.co"; do
  code=$(curl -sL -o /dev/null -w '%{http_code}' --max-time 20 "$candidate/api/models/$MODEL" || echo 000)
  printf '  %-28s %s\n' "$candidate" "$code"
  if [ "$code" = "200" ] && [ -z "$ENDPOINT" ]; then
    ENDPOINT="$candidate"
  fi
done

if [ -z "$ENDPOINT" ]; then
  echo
  echo "no HuggingFace-compatible endpoint answered; trying ModelScope instead"
  if command -v modelscope >/dev/null 2>&1; then
    modelscope download --model "$MODEL" --local_dir "$TARGET"
  else
    echo "FAILED: neither a HF endpoint nor the modelscope CLI is available"
    echo "install one of:  uv pip install huggingface_hub   |   pip install modelscope"
    exit 1
  fi
else
  echo
  echo "  using $ENDPOINT"
  code=$(curl -sL --max-time 25 "$ENDPOINT/api/models/$MODEL" -o /tmp/model-api.json -w '%{http_code}')
  if [ "$code" = "200" ]; then
    python3 - <<'PY' 2>/dev/null || true
import json
data = json.load(open("/tmp/model-api.json"))
print("  resolved revision:", data.get("sha"))
print("  last modified:    ", data.get("lastModified"))
PY
  fi

  export HF_ENDPOINT="$ENDPOINT"
  START=$(date +%s)
  if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL" --local-dir "$TARGET"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$MODEL" --local-dir "$TARGET"
  else
    python3 -m pip install --quiet huggingface_hub
    python3 -m huggingface_hub.commands.huggingface_cli download "$MODEL" --local-dir "$TARGET"
  fi
  END=$(date +%s)
  echo "  download took $((END - START)) seconds"
fi

echo
echo "=== verification against the pinned files ==="
failures=0
for pair in "config.json=$EXPECT_CONFIG" "tokenizer_config.json=$EXPECT_TOKENIZER"; do
  name=${pair%%=*}
  expected=${pair#*=}
  path="$TARGET/$name"
  if [ ! -f "$path" ]; then
    echo "  FAIL  $name missing"
    failures=$((failures + 1))
    continue
  fi
  actual=$(sha256sum "$path" | cut -d' ' -f1)
  if [ "$actual" = "$expected" ]; then
    echo "  ok    $name"
  else
    echo "  FAIL  $name"
    echo "        expected $expected"
    echo "        actual   $actual"
    failures=$((failures + 1))
  fi
done

echo
echo "=== contents ==="
du -sh "$TARGET"
ls -1 "$TARGET" | head -20
echo "  weight shards: $(find "$TARGET" -name '*.safetensors' | wc -l)"

echo
if [ $failures -eq 0 ]; then
  echo "RESULT: OK"
  exit 0
fi
echo "RESULT: FAILED ($failures) - do not train against this copy"
exit 1
