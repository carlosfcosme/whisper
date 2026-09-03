#!/usr/bin/env bash
# Offline system-dep smoke serve. Binds 127.0.0.1 only.
# No model-weight download. No Hub.
set -euo pipefail

cd "$(dirname "$0")/.."

exec python3 scripts/system_deps_smoke.py \
  --serve \
  --host 127.0.0.1 \
  --port "${WHISPER_SERVE_PORT:-8765}"
