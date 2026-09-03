#!/usr/bin/env bash
# Bind 127.0.0.1 only. CPU path. No Hub. No weight download.
set -euo pipefail

cd "$(dirname "$0")/.."

export WHISPER_OFFLINE="${WHISPER_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

exec python3 -m whisper.serve \
  --host 127.0.0.1 \
  --port "${WHISPER_SERVE_PORT:-8765}"
