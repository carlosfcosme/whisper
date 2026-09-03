#!/usr/bin/env bash
# Executable localhost-only test runner.
# Blocks network/model downloads via WHISPER_LOCALHOST_ONLY + pytest hooks.
# Writes caches under a disposable XDG_CACHE_HOME (gitignored artifacts).
set -euo pipefail

cd "$(dirname "$0")/.."

export WHISPER_LOCALHOST_ONLY=1
export WHISPER_DEVICE=cpu
export CUDA_VISIBLE_DEVICES=
export TRANSFORMERS_OFFLINE=1

CACHE_DIR="${XDG_CACHE_HOME:-}"
if [ -z "$CACHE_DIR" ]; then
  CACHE_DIR="$(mktemp -d)"
  cleanup() { rm -rf "$CACHE_DIR"; }
  trap cleanup EXIT
fi
export XDG_CACHE_HOME="$CACHE_DIR"

exec python3 -m pytest -q -m 'localhost_only and not requires_cuda' "$@"
