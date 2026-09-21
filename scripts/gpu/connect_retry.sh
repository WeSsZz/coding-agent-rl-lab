#!/usr/bin/env bash
# Retry the GPU instance connection with diagnostics, because a freshly started AutoDL
# instance can accept TCP before sshd is serving a banner.
#
# Usage: bash scripts/gpu/connect_retry.sh [port] [address] [attempts]
set -uo pipefail

PORT=${1:-40772}
ADDRESS=${2:-106.120.183.117}
ATTEMPTS=${3:-6}
KEY=${AUTODL_KEY:-$HOME/.ssh/id_ed25519_autodl}

for attempt in $(seq 1 "$ATTEMPTS"); do
  echo "=== attempt $attempt/$ATTEMPTS  $(date -u +%FT%TZ) ==="
  output=$(timeout 45 ssh -p "$PORT" -i "$KEY" \
    -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=30 -o ConnectionAttempts=1 \
    "root@$ADDRESS" 'hostname' 2>&1)
  status=$?
  echo "$output" | tail -3
  if [ $status -eq 0 ]; then
    echo "connected after $attempt attempt(s)"
    exit 0
  fi
  if [ $attempt -lt "$ATTEMPTS" ]; then
    echo "  (exit $status; sleeping 20s)"
    sleep 20
  fi
done

echo
echo "=== diagnostics ==="
echo "--- tcp probe ---"
timeout 10 bash -c "cat < /dev/null > /dev/tcp/$ADDRESS/$PORT" 2>&1 && echo "port open" || echo "port closed"
echo "--- banner grab ---"
timeout 20 bash -c "exec 3<>/dev/tcp/$ADDRESS/$PORT; head -c 100 <&3" 2>&1 | head -2 || echo "(no banner within 20s)"
echo "--- verbose ssh, last lines ---"
timeout 40 ssh -vv -p "$PORT" -i "$KEY" \
  -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=25 "root@$ADDRESS" 'true' 2>&1 | tail -12
exit 1
