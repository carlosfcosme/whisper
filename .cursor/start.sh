#!/usr/bin/env bash
# Weights-free helper listener. Bind 127.0.0.1 only.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -m whisper.serve --host 127.0.0.1 --port "${WHISPER_SERVE_PORT:-8765}"
