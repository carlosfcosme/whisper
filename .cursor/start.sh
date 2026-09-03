#!/usr/bin/env bash
# Local health server. Host is hardcoded to loopback.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -m whisper.serve --host 127.0.0.1 --port "${WHISPER_SERVE_PORT:-8765}"
