#!/usr/bin/env bash
# Reach the AutoDL GPU instance from the worker, working around the DNS problem.
#
# `connect.bjb1.seetacloud.com` has previously resolved to a Clash fake-IP (198.18.0.4) on both
# the Windows box and this VM, so the name alone is not reliable. This resolves it through
# several resolvers, tries each candidate address, and reports which one actually answers.
#
# Usage: bash scripts/gpu/connect_probe.sh [port] [host]
set -uo pipefail

PORT=${1:-40772}
HOST=${2:-connect.bjb1.seetacloud.com}
KEY=${AUTODL_KEY:-$HOME/.ssh/id_ed25519_autodl}

echo "=== resolvers ==="
for dns in "" 223.5.5.5 8.8.8.8 1.1.1.1; do
  if [ -z "$dns" ]; then
    printf '  system      '
    getent hosts "$HOST" | awk '{print $1}' | sort -u | tr '\n' ' '
    echo
  else
    printf '  %-11s ' "$dns"
    timeout 8 nslookup "$HOST" "$dns" 2>/dev/null | awk '/^Address: /{print $2}' | grep -v '#' | tr '\n' ' '
    echo
  fi
done

echo
echo "=== key ==="
if [ -f "$KEY" ]; then
  echo "  $KEY present ($(stat -c%s "$KEY") bytes)"
else
  echo "  $KEY MISSING"
  exit 2
fi

echo
echo "=== candidate addresses ==="
CANDIDATES=$( { getent hosts "$HOST" | awk '{print $1}'; echo 106.120.183.117; } | sort -u )

for address in $CANDIDATES; do
  printf '  %-18s ' "$address"
  if timeout 12 bash -c "cat < /dev/null > /dev/tcp/$address/$PORT" 2>/dev/null; then
    echo "port $PORT open"
  else
    echo "port $PORT closed/unreachable"
  fi
done

echo
echo "=== ssh attempt ==="
for address in $CANDIDATES; do
  echo "--- $address:$PORT ---"
  timeout 30 ssh -p "$PORT" -i "$KEY" \
    -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=15 "root@$address" \
    'hostname; echo "--- gpu ---"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; nvidia-smi | grep -i "CUDA Version"; echo "--- disk ---"; df -h / /root/autodl-tmp 2>/dev/null | sort -u' 2>&1 | head -14
  status=$?
  if [ $status -eq 0 ]; then
    echo "  ^ $address works"
    echo "$address" > /tmp/autodl-address.txt
    exit 0
  fi
done

echo "no candidate address answered"
exit 1
