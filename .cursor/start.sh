#!/usr/bin/env bash
# Whisper serve/bind path. Listens on 127.0.0.1 only.
# No model-weight download. No Hub.
set -euo pipefail

cd "$(dirname "$0")/.."

export WHISPER_ALLOW_WEIGHT_FETCH="${WHISPER_ALLOW_WEIGHT_FETCH:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

exec python3 -m whisper.serve \
  --host 127.0.0.1 \
  --port "${WHISPER_SERVE_PORT:-8765}"
