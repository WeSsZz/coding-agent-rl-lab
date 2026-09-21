#!/usr/bin/env bash
# Determine where Qwen3-8B can actually be fetched from, and pin its revision.
#
# The model is not pre-installed anywhere in this project: the old instance carries
# Qwen2.5-Coder snapshots from route v1, and route v2 uses Qwen3-8B, which is ~16 GB of
# bfloat16 safetensors that must be downloaded.
#
# `main` is a moving target, so this records the revision sha the model resolved to before any
# weights are fetched. That sha, plus the sha256 of config.json and tokenizer_config.json
# recorded in envs/qwen3-8b-chat-template.manifest.json, is what makes "we ran this model"
# checkable later instead of merely asserted.
#
# Usage: bash scripts/gpu/probe_model_sources.sh
set -uo pipefail

MODEL=Qwen/Qwen3-8B

probe() {
  local name=$1 url=$2
  local code
  code=$(curl -sL -o /tmp/probe.out -w '%{http_code}' --max-time 25 "$url")
  printf '  %-46s %s\n' "$name" "$code"
  return 0
}

echo "=== host reachability ==="
probe "huggingface.co"           "https://huggingface.co/"
probe "hf-mirror.com"            "https://hf-mirror.com/"
probe "modelscope.cn"            "https://www.modelscope.cn/"
probe "www.modelscope.cn api"    "https://www.modelscope.cn/api/v1/models/${MODEL}"

echo
echo "=== pinned revision, via whichever host answers ==="
for endpoint in "https://hf-mirror.com" "https://huggingface.co"; do
  printf '  %s -> ' "$endpoint"
  if curl -sL --max-time 25 "$endpoint/api/models/${MODEL}" -o /tmp/model-api.json 2>/dev/null; then
    if command -v python3 >/dev/null 2>&1; then
      python3 - <<'PY' 2>/dev/null || echo "(could not parse)"
import json
try:
    data = json.load(open("/tmp/model-api.json"))
except Exception as exc:
    print("unparseable:", exc)
    raise SystemExit(0)
print("sha=", data.get("sha"), "lastModified=", data.get("lastModified"),
      "siblings=", len(data.get("siblings", [])))
PY
    else
      head -c 200 /tmp/model-api.json
      echo
    fi
    break
  fi
  echo "(no answer)"
done

echo
echo "=== files this project already pinned, for cross-checking a download ==="
echo "  envs/qwen3-8b-tokenizer_config.json"
echo "  envs/qwen3-8b-chat-template.manifest.json"
if command -v sha256sum >/dev/null 2>&1; then
  for candidate in /tmp/q3-tok.json /tmp/q3-config.json; do
    [ -f "$candidate" ] && sha256sum "$candidate"
  done
fi

echo
echo "=== expected download size (approx) ==="
echo "  Qwen3-8B bfloat16 safetensors: ~16 GB (8.2B params x 2 bytes) + tokenizer"
echo
echo "=== how to fetch, in order of preference on a China-egress box ==="
echo "  1. HF_ENDPOINT=https://hf-mirror.com hf download ${MODEL} --local-dir <dir>"
echo "  2. modelscope download --model ${MODEL} --local_dir <dir>"
echo "  3. huggingface-cli download ${MODEL} --local-dir <dir>   (needs direct HF access)"
