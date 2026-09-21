#!/usr/bin/env bash
# Syntax-check every shell script this project ships for the GPU box and the worker.
# `bash -n` catches quoting and structure errors that only appear when the script runs, and the
# GPU box is the worst place to discover one.
#
# Usage: bash scripts/check_shell_syntax.sh [repo-root]
set -uo pipefail

if [ $# -ge 1 ]; then
  REPO_ROOT=$(cd -- "$1" && pwd)
else
  HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  REPO_ROOT=$(cd -- "$HERE/.." && pwd)
fi

echo "repo root: $REPO_ROOT"

failures=0
checked=0

while IFS= read -r script; do
  checked=$((checked + 1))
  printf '%-60s ' "${script#"$REPO_ROOT"/}"
  if bash -n "$script" 2>/tmp/syntax-error.txt; then
    echo "ok"
  else
    echo "FAIL"
    sed 's/^/    /' /tmp/syntax-error.txt
    failures=$((failures + 1))
  fi
done < <(find "$REPO_ROOT/scripts" -name '*.sh' -type f | sort)

echo
echo "checked $checked script(s), $failures failure(s)"
[ $failures -eq 0 ]
