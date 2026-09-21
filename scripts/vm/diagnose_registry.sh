#!/usr/bin/env bash
# Diagnose why docker pulls fail on the worker: DNS first, then registry mirrors.
set -uo pipefail

echo "=== resolv.conf ==="
cat /etc/resolv.conf 2>/dev/null | sed -n '1,12p'

echo "=== resolution of registry hosts ==="
for host in registry-1.docker.io auth.docker.io production.cloudflare.docker.com; do
  printf '%-40s ' "$host"
  getent hosts "$host" | head -2 | tr '\n' ' '
  echo
done

echo "=== direct TCP to the registry ==="
timeout 12 curl -s -o /dev/null -w 'registry-1.docker.io %{http_code} %{time_total}s\n' https://registry-1.docker.io/v2/ || echo 'registry-1.docker.io FAILED'

echo "=== candidate registry mirrors ==="
for mirror in \
  https://docker.m.daocloud.io/v2/ \
  https://dockerproxy.net/v2/ \
  https://docker.1panel.live/v2/ \
  https://hub-mirror.c.163.com/v2/ \
  https://mirror.ccs.tencentyun.com/v2/ ; do
  printf '%-42s ' "$mirror"
  timeout 12 curl -s -o /dev/null -w '%{http_code} %{time_total}s\n' "$mirror" || echo FAILED
done

echo "=== daemon config ==="
cat /etc/docker/daemon.json 2>/dev/null || echo '(no /etc/docker/daemon.json)'

echo "=== public DNS reachability ==="
for dns in 223.5.5.5 8.8.8.8 1.1.1.1; do
  printf '%-10s ' "$dns"
  timeout 8 getent hosts one.one.one.one >/dev/null 2>&1 && echo ok || echo 'no (irrelevant)'
  timeout 8 nslookup registry-1.docker.io "$dns" 2>/dev/null | sed -n '/^Address: /p' | tail -2 | tr '\n' ' '
  echo
done
