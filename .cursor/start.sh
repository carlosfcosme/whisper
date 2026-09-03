#!/usr/bin/env bash
# Whisper serve/bind path. Listens on 127.0.0.1 only.
# No model-weight download. No Hub. See scripts/check_no_wildcard_bind.py.
set -euo pipefail

cd "$(dirname "$0")/.."

export WHISPER_OFFLINE=1

exec python3 -m whisper.serve \
  --host 127.0.0.1 \
  --port "${WHISPER_SERVE_PORT:-8765}"
