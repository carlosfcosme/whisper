#!/usr/bin/env bash
# Weights-free health server on 127.0.0.1 only. Does not call load_model.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -m whisper.serve --host 127.0.0.1 --port "${WHISPER_PORT:-8765}"
