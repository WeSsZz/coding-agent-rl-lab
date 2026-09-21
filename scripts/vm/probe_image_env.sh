#!/usr/bin/env bash
# Inspect a SWE-Gym task image: where the grading interpreter lives, where the repo is,
# and what actually happens when a declared FAIL_TO_PASS node is handed to pytest.
#
# Usage: bash scripts/vm/probe_image_env.sh <image> [node_id]
set -uo pipefail

IMAGE=${1:?usage: probe_image_env.sh <image> [node_id]}
NODE=${2:-}

echo "=== image: $IMAGE ==="
docker run --rm --platform linux/amd64 --network none --entrypoint bash "$IMAGE" -lc '
  echo "--- conda envs ---"
  ls /opt/miniconda3/envs/ 2>/dev/null || echo "(no /opt/miniconda3)"
  echo "--- interpreters on PATH ---"
  command -v python python3 pytest || true
  echo "--- testbed root ---"
  ls -d /testbed 2>/dev/null && (cd /testbed && git rev-parse HEAD && git log -1 --format=%s | cut -c1-70)
  echo "--- testbed top level ---"
  ls /testbed 2>/dev/null | head -20
  echo "--- pytest versions found ---"
  for candidate in /opt/miniconda3/envs/testbed/bin/pytest /opt/miniconda3/bin/pytest; do
    if [ -x "$candidate" ]; then
      printf "%s -> " "$candidate"
      "$candidate" --version 2>&1 | head -1
    fi
  done
  echo "--- pytest config in repo ---"
  for cfg in pytest.ini setup.cfg pyproject.toml tox.ini; do
    if [ -f "/testbed/$cfg" ]; then
      echo "[$cfg]"
      grep -n -A6 -E "^\[tool\.pytest|^\[pytest\]|^\[tool:pytest\]" "/testbed/$cfg" | head -20
    fi
  done
'

if [ -n "$NODE" ]; then
  echo
  echo "=== running one declared node: $NODE ==="
  docker run --rm --platform linux/amd64 --network none --entrypoint bash "$IMAGE" -lc \
    "cd /testbed && timeout 300 /opt/miniconda3/envs/testbed/bin/pytest -n0 -rA -- '$NODE' 2>&1 | tail -25; echo EXIT=\${PIPESTATUS[0]}"
  echo
  echo "=== same node, python -m pytest ==="
  docker run --rm --platform linux/amd64 --network none --entrypoint bash "$IMAGE" -lc \
    "cd /testbed && timeout 300 /opt/miniconda3/envs/testbed/bin/python -m pytest -rA --tb=no -p no:cacheprovider -- '$NODE' 2>&1 | tail -25; echo EXIT=\${PIPESTATUS[0]}"
fi
