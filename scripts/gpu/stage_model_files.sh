#!/usr/bin/env bash
# Pin a model's chat template the same way regardless of which model is chosen.
#
# Writes the tokenizer_config.json, config.json, the resolved revision and their sha256 values
# into a staging directory, so the repository can pin the template and record integrity anchors
# without depending on the model API being reachable later.
#
# Usage: bash scripts/gpu/stage_model_files.sh <model-id> [output-dir]
set -uo pipefail

MODEL=${1:?usage: stage_model_files.sh <model-id> [output-dir]}
OUT=${2:-/tmp/staged-model}
ENDPOINT=${ENDPOINT:-https://hf-mirror.com}
mkdir -p "$OUT"

safe=$(printf '%s' "$MODEL" | tr -c 'a-zA-Z0-9' '_')

echo "model:    $MODEL"
echo "endpoint: $ENDPOINT"
echo "staging:  $OUT"
echo

# Revision first, with retries: the mirror drops connections intermittently.
REVISION=""
for attempt in 1 2 3 4; do
  code=$(curl -sL --max-time 30 -o "$OUT/api.json" -w '%{http_code}' \
         "$ENDPOINT/api/models/$MODEL" 2>/dev/null) || code=000
  if [ "$code" = "200" ]; then
    REVISION=$(python3 -c "import json;print(json.load(open('$OUT/api.json')).get('sha',''))" 2>/dev/null)
    [ -n "$REVISION" ] && break
  fi
  echo "  revision lookup attempt $attempt failed (http $code)"
  sleep 4
done

if [ -z "$REVISION" ]; then
  echo "FAILED: could not resolve a revision; retry rather than concluding the model is gone"
  exit 1
fi
echo "revision: $REVISION"
echo

for name in tokenizer_config.json config.json; do
  ok=0
  for attempt in 1 2 3 4; do
    code=$(curl -sL --max-time 60 -o "$OUT/$safe.$name" -w '%{http_code}' \
           "$ENDPOINT/$MODEL/resolve/$REVISION/$name" 2>/dev/null) || code=000
    if [ "$code" = "200" ] && python3 -c "import json,sys;json.load(open('$OUT/$safe.$name'))" 2>/dev/null; then
      ok=1
      break
    fi
    echo "  $name attempt $attempt failed (http $code)"
    sleep 4
  done
  printf '%-24s ' "$name"
  if [ $ok -eq 1 ]; then
    echo "sha256 $(sha256sum "$OUT/$safe.$name" | awk '{print $1}')  $(wc -c < "$OUT/$safe.$name") bytes"
  else
    echo "FAILED"
  fi
done

echo
echo "=== files staged in $OUT ==="
ls -1 "$OUT" | grep "^$safe" || true
