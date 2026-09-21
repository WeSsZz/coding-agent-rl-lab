#!/usr/bin/env bash
# Check which task images actually exist upstream, without pulling them.
#
# The official SWE-Gym data includes instances whose instance image was never published
# (upstream reports 38 such rows, spread across pandas, dvc, conan, dask, modin, hydra,
# pydantic, bokeh and monai). A split that contains one of them is not a runnable split: the
# container start fails, and if that happens mid-rollout it is an infrastructure failure rather
# than a policy failure, which is exactly the confusion this project keeps trying to avoid.
#
# `docker manifest inspect` talks to the registry directly and does not need the local daemon,
# so it works even though `registry-1.docker.io` is unreachable here.
#
# Usage:
#   bash scripts/vm/check_images_exist.sh <image-list-file> [output-missing-file]
set -uo pipefail

LIST=${1:?usage: check_images_exist.sh <image-list-file> [output-missing-file]}
MISSING_OUT=${2:-}

MIRRORS=(
  "dockerproxy.net"
  "docker.1panel.live"
)
# `docker.m.daocloud.io` is deliberately absent: it answers `docker.io/xingyaoww/* is
# forbidden`, so it can neither confirm nor deny, and it hangs long enough to dominate the run.

mapfile -t IMAGES < <(grep -v '^[[:space:]]*$' "$LIST")
echo "checking ${#IMAGES[@]} images"

available=0
missing=0
unknown=0
missing_images=()
unknown_images=()

for image in "${IMAGES[@]}"; do
  repo=${image%%:*}
  found=0
  saw_error=0
  for mirror in "${MIRRORS[@]}"; do
    output=$(timeout 25 docker manifest inspect "$mirror/$repo:latest" 2>&1)
    status=$?
    if [ $status -eq 0 ]; then
      found=1
      break
    fi
    case "$output" in
      *"not found"*|*"manifest unknown"*|*"NAME_UNKNOWN"*)
        # This mirror does not have it; another one may.
        ;;
      *)
        saw_error=1
        ;;
    esac
  done
  if [ $found -eq 1 ]; then
    available=$((available + 1))
  elif [ $saw_error -eq 1 ]; then
    echo "UNKNOWN   $image"
    unknown=$((unknown + 1))
    unknown_images+=("$image")
  else
    echo "MISSING   $image"
    missing=$((missing + 1))
    missing_images+=("$image")
  fi
done

echo
echo "=== available=$available missing=$missing unknown=$unknown ==="

if [ -n "$MISSING_OUT" ]; then
  printf '%s\n' "${missing_images[@]:-}" | grep -v '^$' > "$MISSING_OUT" || true
  echo "wrote ${#missing_images[@]} missing image name(s) to $MISSING_OUT"
fi

# A registry hiccup is not evidence that an image is absent; only a clean "not found" is.
[ $unknown -eq 0 ]
