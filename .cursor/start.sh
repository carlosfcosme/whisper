#!/usr/bin/env bash
# Weights-free health server. Bind 127.0.0.1 only (never all-interfaces).
set -euo pipefail
cd "$(dirname "$0")/.."
port="${WHISPER_SERVE_PORT:-8765}"
exec python3 -m whisper.serve --host 127.0.0.1 --port "${port}"
