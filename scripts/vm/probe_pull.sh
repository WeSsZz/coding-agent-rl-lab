#!/usr/bin/env bash
# Try one image against every mirror with a short timeout, and report what each says.
# Usage: bash scripts/vm/probe_pull.sh <image>
set -uo pipefail

IMAGE=${1:?usage: probe_pull.sh <image>}
REPO=${IMAGE%%:*}
MIRRORS=(
  "dockerproxy.net"
  "docker.1panel.live"
  "docker.m.daocloud.io"
)

for mirror in "${MIRRORS[@]}"; do
  printf '=== %s/%s ===\n' "$mirror" "$REPO"
  start=$(date +%s)
  if timeout 180 docker pull "$mirror/$REPO:latest" > /tmp/probe-pull.log 2>&1; then
    echo "  PULL OK in $(( $(date +%s) - start ))s"
    docker tag "$mirror/$REPO:latest" "$IMAGE"
    echo "  tagged -> $IMAGE"
    exit 0
  fi
  echo "  failed after $(( $(date +%s) - start ))s"
  tail -4 /tmp/probe-pull.log | sed 's/^/    /'
done
echo "no mirror served $REPO"
exit 1
