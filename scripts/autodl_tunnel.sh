#!/bin/sh
# Keep a local port forwarded to the model host on the GPU box.
#
# The rollout host is reached from the VM only, and its SSH tunnel dies independently of the
# instance: the first v24 arm lost two hours to a tunnel that closed two minutes in, because
# every later request failed and the rollouts recorded the failures one step at a time. A
# reconnect loop costs nothing and removes that failure mode; `require_served_model` in
# swe_gym_rollout refuses to start a run against a port this loop has not restored.
#
# Usage: sh scripts/autodl_tunnel.sh [port] [host]
set -u
PORT=${1:-10283}
HOST=${2:-connect.bjb1.seetacloud.com}
KEY=${AUTODL_KEY:-$HOME/.ssh/id_ed25519_autodl}
LOG=${AUTODL_TUNNEL_LOG:-/tmp/autodl_tunnel.log}

while true; do
  ssh -N -L "127.0.0.1:8000:127.0.0.1:8000" -p "$PORT" -i "$KEY" \
    -o BatchMode=yes -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    "root@$HOST"
  echo "tunnel exited at $(date -u +%FT%TZ), retrying" >> "$LOG"
  sleep 5
done
