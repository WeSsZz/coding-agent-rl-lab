#!/usr/bin/env bash
# Pull task images through a reachable mirror and retag them to the canonical name.
#
# `registry-1.docker.io` is DNS-poisoned on this network (it answers with Facebook address
# ranges), and the daemon-wide mirror setting needs root, which this account does not have.
# Pulling `<mirror>/<repo>` and retagging needs no privileges and leaves an image with exactly
# the name the environment code expects.
#
# Pre-pulling is mandatory, not an optimisation: a container start cannot fetch a missing image
# here, so a missing image would surface as an infrastructure failure in the middle of a rollout.
#
# Usage:
#   bash scripts/vm/pull_task_images.sh <image-list-file>        # one image per line
#   bash scripts/vm/pull_task_images.sh <image> [image ...]      # explicit names
#
# The image list for the frozen split is produced by:
#   python scripts/build_task_splits.py --image-list work/private/split-images.txt
set -uo pipefail

if [ $# -eq 0 ]; then
  echo "usage: pull_task_images.sh <image-list-file> | <image> [image ...]" >&2
  exit 2
fi

if [ $# -eq 1 ] && [ -f "$1" ]; then
  mapfile -t IMAGES < <(grep -v '^[[:space:]]*$' "$1")
else
  IMAGES=("$@")
fi

MIRRORS=(
  "dockerproxy.net"
  "docker.1panel.live"
  "docker.m.daocloud.io"
)

strip_tag() {
  # xingyaoww/name:latest -> xingyaoww/name
  echo "${1%%:*}"
}

echo "=== images requested: ${#IMAGES[@]} ==="
echo "=== disk before ==="
df -h / | tail -1

ok=0
skipped=0
failed=()

for target in "${IMAGES[@]}"; do
  if docker image inspect "$target" >/dev/null 2>&1; then
    skipped=$((skipped + 1))
    continue
  fi
  repo=$(strip_tag "$target")
  pulled=0
  for mirror in "${MIRRORS[@]}"; do
    if timeout 1800 docker pull "$mirror/$repo:latest" >/tmp/pull.log 2>&1; then
      docker tag "$mirror/$repo:latest" "$target"
      echo "ok      $target"
      pulled=1
      ok=$((ok + 1))
      break
    fi
  done
  if [ "$pulled" -eq 0 ]; then
    echo "FAILED  $target  ($(tail -1 /tmp/pull.log | cut -c1-100))"
    failed+=("$target")
  fi
done

echo
echo "=== summary: pulled=$ok already_present=$skipped failed=${#failed[@]} ==="
for item in "${failed[@]:-}"; do
  [ -n "$item" ] && echo "  could not pull: $item"
done

echo "=== disk after ==="
df -h / | tail -1
echo "=== resolved image ids (record these: mirror pulls are not digest-verified) ==="
for target in "${IMAGES[@]}"; do
  docker image inspect "$target" --format '{{.RepoTags}} {{.Id}}' 2>/dev/null || echo "$target MISSING"
done

[ ${#failed[@]} -eq 0 ]
