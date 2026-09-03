#!/usr/bin/env bash
# Loopback-only health server. Never bind anything other than 127.0.0.1.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -m whisper.serve --host 127.0.0.1 --port "${WHISPER_PORT:-8765}"
