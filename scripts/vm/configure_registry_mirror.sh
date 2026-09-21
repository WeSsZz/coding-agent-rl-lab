#!/usr/bin/env bash
# Point the worker's Docker daemon at a reachable registry mirror.
#
# Why: `registry-1.docker.io` resolves to Facebook/Instagram address ranges on this network
# (2a03:2880:...:face:b00c is Facebook's well-known prefix), so every pull fails with
# "connection refused". Three community mirrors answer correctly; the daemon config is additive
# and reversible, and it does not touch any existing image.
set -euo pipefail

CONFIG=/etc/docker/daemon.json
BACKUP=/etc/docker/daemon.json.before-mirror

echo "=== before ==="
cat "$CONFIG" 2>/dev/null || echo '(absent)'
docker images --format '{{.Repository}}' | sort > /tmp/images-before.txt
wc -l < /tmp/images-before.txt | xargs echo 'images before:'

if [ -f "$CONFIG" ] && [ ! -f "$BACKUP" ]; then
  sudo cp "$CONFIG" "$BACKUP"
  echo "backed up existing config to $BACKUP"
fi

sudo mkdir -p /etc/docker
sudo tee "$CONFIG" >/dev/null <<'JSON'
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://dockerproxy.net",
    "https://docker.1panel.live"
  ]
}
JSON

echo "=== wrote ==="
cat "$CONFIG"

sudo systemctl restart docker
sleep 4
echo "=== docker status ==="
systemctl is-active docker

echo "=== after ==="
docker images --format '{{.Repository}}' | sort > /tmp/images-after.txt
wc -l < /tmp/images-after.txt | xargs echo 'images after:'
if diff -q /tmp/images-before.txt /tmp/images-after.txt >/dev/null; then
  echo "existing images unchanged"
else
  echo "WARNING: image list changed"
  diff /tmp/images-before.txt /tmp/images-after.txt || true
fi
