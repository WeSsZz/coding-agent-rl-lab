#!/usr/bin/env bash
set -euo pipefail

# CPython 3.12's linker flags include a GNU ld option unsupported by Zig 0.13.
# Filter that one option while preserving all compiler and linker inputs.
args=()
for arg in "$@"; do
  if [[ "$arg" != '-Wl,--exclude-libs,ALL' ]]; then
    args+=("$arg")
  fi
done
exec "$HOME/.local/zig/zig" c++ "${args[@]}"
