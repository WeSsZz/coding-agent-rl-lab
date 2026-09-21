#!/usr/bin/env bash
# Fetch a few files from the pinned veRL revision so install instructions are read, not guessed.
set -uo pipefail
REV=1876b06d0a3e4e71e06230be10af14492ca8a75b
BASE=https://raw.githubusercontent.com/verl-project/verl/$REV
OUT=${1:-/tmp/verl}
mkdir -p "$OUT"

for path in docs/start/install.rst README.md manage_envs.py pyproject.toml; do
  name=$(echo "$path" | tr '/' '_')
  printf '%-28s ' "$path"
  code=$(curl -sL --max-time 40 -o "$OUT/$name" -w '%{http_code}' "$BASE/$path")
  printf '%s %s bytes\n' "$code" "$(wc -c < "$OUT/$name")"
done

echo
echo "=== install.rst, first 80 lines ==="
sed -n '1,80p' "$OUT/docs_start_install.rst" 2>/dev/null || echo "(missing)"

echo
echo "=== README, install section ==="
grep -n -i -A12 'installation\|pip install\|uv sync' "$OUT/README.md" 2>/dev/null | head -60
