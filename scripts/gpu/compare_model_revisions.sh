#!/usr/bin/env bash
# Compare the chat template served by `main` against the one served by an explicit revision.
#
# Why: the config files this repository pinned were fetched from resolve/main through a mirror,
# and mirrors cache that path. If `main` served a stale copy, then the pinned template belongs to
# an older revision than the one the model API reports, and every prompt rendered from it would
# differ from the model actually under test. That is exactly the kind of silent mismatch this
# project exists to avoid.
#
# Usage: bash scripts/gpu/compare_model_revisions.sh [model-id] [revision] [endpoint]
set -uo pipefail

MODEL=${1:-Qwen/Qwen3-8B}
REVISION=${2:-b968826d9c46dd6066d109eabc6255188de91218}
ENDPOINT=${3:-https://hf-mirror.com}
WORK=${WORK:-/tmp/model-revision-compare}
mkdir -p "$WORK"

fetch() {
  local ref=$1 name=$2 out=$3
  curl -sL --max-time 120 -o "$out" -w '%{http_code}' "$ENDPOINT/$MODEL/resolve/$ref/$name"
}

echo "model:    $MODEL"
echo "endpoint: $ENDPOINT"
echo "work:     $WORK"
echo

for ref in main "$REVISION"; do
  # printf, not echo: `echo` appends a newline, and `tr -c` would turn that newline into a
  # trailing underscore, so the shell and the Python below would disagree about the filename.
  safe=$(printf '%s' "$ref" | tr -c 'a-zA-Z0-9' '_')
  code=$(fetch "$ref" tokenizer_config.json "$WORK/tok-$safe.json")
  size=$(wc -c < "$WORK/tok-$safe.json")
  printf 'tokenizer_config.json @ %-42s http=%s %s bytes\n' "$ref" "$code" "$size"
  printf '  sha256 %s\n' "$(sha256sum "$WORK/tok-$safe.json" | awk '{print $1}')"
done

python3 - "$WORK" main "$REVISION" <<'PY'
import json, pathlib, sys

work, ref_a, ref_b = sys.argv[1], sys.argv[2], sys.argv[3]

def load(ref):
    safe = "".join(c if c.isalnum() else "_" for c in ref)
    path = pathlib.Path(work) / f"tok-{safe}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  could not read {ref}: {exc}")
        return None

a, b = load(ref_a), load(ref_b)
if a is None or b is None:
    sys.exit(2)

ta, tb = a.get("chat_template"), b.get("chat_template")
print()
print("chat_template lengths:", len(ta or ""), "vs", len(tb or ""))
print("identical:", ta == tb)

if ta != tb:
    import difflib
    print()
    print("--- diff (main -> revision) ---")
    for line in difflib.unified_diff(
        (ta or "").splitlines(), (tb or "").splitlines(),
        fromfile=ref_a, tofile=ref_b, lineterm="", n=1,
    ):
        print("  " + line[:160])

for key in ("tokenizer_class", "model_max_length", "eos_token", "pad_token"):
    print(f"{key:18s} {a.get(key)!r}  vs  {b.get(key)!r}")
PY
