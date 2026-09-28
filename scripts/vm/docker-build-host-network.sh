#!/usr/bin/env bash
set -euo pipefail

# This VM's Docker bridge cannot resolve DNS. Only the image *build* uses
# host networking; OpenHands task containers keep the fork's network setting.
if [[ "${1:-}" == buildx && "${2:-}" == build ]]; then
  shift 2
  build_context="${@: -1}"
  if [[ -d "$build_context" ]]; then
    cp -a "$build_context" "$HOME/swegym-runtime-context-$(date +%s)"
    python3 "$HOME/prepare_runtime_build_context.py" "$build_context"
  fi
  set +e
  /usr/bin/docker buildx build --network=host "$@" 2>&1 | tee "$HOME/swegym-docker-build.log"
  build_status=${PIPESTATUS[0]}
  exit "$build_status"
fi
exec /usr/bin/docker "$@"
