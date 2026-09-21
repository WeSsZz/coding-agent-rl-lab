#!/usr/bin/env bash
# Copy a prepared bundle from the Ubuntu worker to the GPU instance.
#
# The Windows editing box has no direct internet route, and the GPU box is only reachable from
# the worker, so deployment is a two hop: Windows -> worker -> instance. This is the second hop.
#
# Usage (on the worker):
#   bash scripts/gpu/deploy_bundle.sh <bundle.tar> <address> <port> [remote-dir]
set -uo pipefail

BUNDLE=${1:?usage: deploy_bundle.sh <bundle.tar> <address> <port> [remote-dir]}
ADDRESS=${2:?missing address}
PORT=${3:?missing port}
REMOTE_DIR=${4:-/root/route-v2-src}
KEY=${AUTODL_KEY:-$HOME/.ssh/id_ed25519_autodl}

if [ ! -f "$BUNDLE" ]; then
  echo "bundle not found: $BUNDLE" >&2
  exit 2
fi

SSH_OPTS=(-p "$PORT" -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=no
          -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20)

echo "=== bundle ==="
ls -lh "$BUNDLE"

echo
echo "=== copying to $ADDRESS:$REMOTE_DIR ==="
timeout 300 scp "${SSH_OPTS[@]}" "$BUNDLE" "root@$ADDRESS:/tmp/route-v2-bundle.tar"
if [ $? -ne 0 ]; then
  echo "scp failed" >&2
  exit 1
fi

echo
echo "=== extracting ==="
timeout 180 ssh "${SSH_OPTS[@]}" "root@$ADDRESS" \
  "rm -rf '$REMOTE_DIR' && mkdir -p '$REMOTE_DIR' && tar -xf /tmp/route-v2-bundle.tar -C '$REMOTE_DIR' && sed -i 's/\r\$//' '$REMOTE_DIR'/scripts/*.sh '$REMOTE_DIR'/scripts/gpu/*.sh '$REMOTE_DIR'/scripts/vm/*.sh 2>/dev/null; ls '$REMOTE_DIR'"
if [ $? -ne 0 ]; then
  echo "extract failed" >&2
  exit 1
fi

echo
echo "=== sanity check on the instance ==="
timeout 180 ssh "${SSH_OPTS[@]}" "root@$ADDRESS" \
  "cd '$REMOTE_DIR' && bash scripts/check_shell_syntax.sh . | tail -3 && python3 -V && (nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo 'no nvidia-smi')"
