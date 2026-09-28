#!/usr/bin/env bash
# Does a model's own chat template support tools at all?
#
# This matters because route v2 adopts the OpenHands function-calling protocol verbatim, which
# requires the template to render a `tools` block and `<tool_call>` / `<tool_response>` markers.
# Route v1 needed a hand-written response template and a `--bare-json-tool-calls` flag for
# Qwen2.5-Coder, which is circumstantial evidence that its template does not do this natively.
# That is worth checking rather than remembering.
#
# Usage: bash scripts/gpu/probe_template_tools.sh [model-id ...]
set -uo pipefail

ENDPOINT=${ENDPOINT:-https://hf-mirror.com}
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
    "Qwen/Qwen2.5-Coder-14B-Instruct"
  )
fi

WORK=${WORK:-/tmp/template-probe}
mkdir -p "$WORK"

for model in "${MODELS[@]}"; do
  safe=$(printf '%s' "$model" | tr -c 'a-zA-Z0-9' '_')
  path="$WORK/$safe.json"

  code=""
  for attempt in 1 2 3; do
    code=$(curl -sL --max-time 40 -o "$path" -w '%{http_code}' \
           "$ENDPOINT/$model/resolve/main/tokenizer_config.json" 2>/dev/null) || code=000
    [ "$code" = "200" ] && break
    sleep 3
  done

  printf '%-38s http=%-4s ' "$model" "$code"
  if [ "$code" != "200" ]; then
    echo "FETCH FAILED"
    continue
  fi

  python3 - "$path" <<'PY'
import json, sys, pathlib

path = pathlib.Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"unparseable ({type(exc).__name__})")
    raise SystemExit(0)

template = data.get("chat_template") or ""
markers = {
    "tools_branch": "{%- if tools %}" in template or "{% if tools %}" in template,
    "tool_call": "<tool_call>" in template,
    "tool_response": "<tool_response>" in template,
    "enable_thinking": "enable_thinking" in template,
}
flags = " ".join(f"{name}={'Y' if value else 'N'}" for name, value in markers.items())
supports = markers["tools_branch"] and markers["tool_call"] and markers["tool_response"]
print(f"{len(template):5d} chars  {flags}   native_tools={'YES' if supports else 'NO'}")
PY
done

echo
echo "native_tools=YES means the OpenHands function-calling schema can be rendered as-is."
echo "native_tools=NO means adopting that protocol on this model needs a bespoke rendering path,"
echo "which is the thing this project decided not to invent."
