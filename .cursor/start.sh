#!/usr/bin/env bash
# Cloud Agent / local start script: whisper serve on 127.0.0.1 only.
# Do not bind all interfaces. Host is hardcoded to loopback.
set -euo pipefail

cd "$(dirname "$0")/.."

port="${WHISPER_SERVE_PORT:-8765}"

exec python3 -m whisper.serve --host 127.0.0.1 --port "${port}"
