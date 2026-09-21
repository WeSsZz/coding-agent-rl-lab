#!/usr/bin/env bash
# Run the SWE-Gym test-command audit inside a freshly extracted copy of the repository.
#
# Kept as a script because the command mixes options that start with '-' with quoted values,
# which does not survive the PowerShell -> ssh -> sh quoting chain.
#
# Usage:
#   bash scripts/vm/run_test_command_audit.sh <bundle.tar> <rows.jsonl> <output.json> <prefix> [prefix ...]
#
# Each prefix is either 'repo/version=COMMAND' or a bare default COMMAND. Scoping is necessary:
# images differ in which plugins are installed, so a flag like '-n0' (xdist) makes pytest exit 4
# on an image without xdist, and an image that ships pytest-pretty replaces the `FAILED <node>`
# summary this project's verifier parses.
set -uo pipefail

BUNDLE=${1:?usage: run_test_command_audit.sh <bundle.tar> <rows.jsonl> <output.json> <prefix> [prefix ...]}
ROWS=${2:?missing rows.jsonl}
OUTPUT=${3:?missing output path}
shift 3
PREFIXES=("$@")
if [ ${#PREFIXES[@]} -eq 0 ]; then
  echo "missing at least one pytest prefix" >&2
  exit 2
fi

STAMP=$(date +%Y%m%d-%H%M%S)
ROOT=/tmp/audit-$STAMP
mkdir -p "$ROOT"
tar -xf "$BUNDLE" -C "$ROOT"
echo "=== extracted to $ROOT ==="
cd "$ROOT" || exit 1

PREFIX_ARGS=()
for prefix in "${PREFIXES[@]}"; do
  PREFIX_ARGS+=(--prefix "$prefix")
done

echo "=== audit ==="
PYTHONPATH=src python3 scripts/audit_swe_gym_test_command.py \
  --rows "$ROWS" \
  --output "$OUTPUT" \
  --test-timeout-seconds 600 \
  "${PREFIX_ARGS[@]}"
STATUS=$?

echo "=== audit exit status: $STATUS ==="
echo "=== leftover containers ==="
docker ps -a --format '{{.Names}} {{.Status}}' | grep coding-agent || echo '(none)'
exit $STATUS
