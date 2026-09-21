#!/usr/bin/env bash
# Pull task images through a reachable mirror and retag them to the canonical name.
#
# `registry-1.docker.io` is DNS-poisoned on this network, and the daemon-wide mirror setting
# needs root, which this account does not have. Pulling `<mirror>/<repo>` and retagging needs
# no privileges and leaves an image with exactly the name the environment code expects.
set -uo pipefail

CANONICAL_REPOS=(
  "xingyaoww/sweb.eval.x86_64.conan-io_s_conan-10975"
  "xingyaoww/sweb.eval.x86_64.conan-io_s_conan-13757"
  "xingyaoww/sweb.eval.x86_64.pydantic_s_pydantic-8947"
)
MIRRORS=(
  "docker.m.daocloud.io"
  "dockerproxy.net"
  "docker.1panel.live"
)

echo "=== disk before ==="
df -h / | tail -1

for repo in "${CANONICAL_REPOS[@]}"; do
  target="$repo:latest"
  if docker image inspect "$target" >/dev/null 2>&1; then
    echo "$target already present"
    continue
  fi
  pulled=0
  for mirror in "${MIRRORS[@]}"; do
    printf 'try %s/%s ... ' "$mirror" "$repo"
    if timeout 900 docker pull "$mirror/$repo:latest" >/tmp/pull.log 2>&1; then
      docker tag "$mirror/$repo:latest" "$target"
      echo "ok"
      pulled=1
      break
    else
      echo "failed ($(tail -1 /tmp/pull.log | cut -c1-90))"
    fi
  done
  if [ "$pulled" -eq 0 ]; then
    echo "COULD NOT PULL $repo from any mirror"
  fi
done

echo "=== disk after ==="
df -h / | tail -1
echo "=== resolved image ids (record these: mirror pulls are not digest-verified) ==="
for repo in "${CANONICAL_REPOS[@]}"; do
  docker image inspect "$repo:latest" --format '{{.RepoTags}} {{index .RepoDigests 0}} id={{.Id}}' 2>/dev/null \
    || echo "$repo:latest MISSING"
done
