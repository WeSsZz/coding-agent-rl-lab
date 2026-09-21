#!/usr/bin/env bash
# Summarise the partial result of a check_images_exist.sh run.
set -uo pipefail
LOG=${1:-/tmp/checkimg.log}
echo "=== log tail ==="
tail -n 12 "$LOG"
echo "=== counts ==="
printf 'UNKNOWN %s\n' "$(grep -c UNKNOWN "$LOG" || true)"
printf 'MISSING %s\n' "$(grep -c MISSING "$LOG" || true)"
echo "=== unknown images ==="
grep UNKNOWN "$LOG" || echo '(none)'
echo "=== missing images ==="
grep MISSING "$LOG" || echo '(none)'
