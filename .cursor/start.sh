#!/usr/bin/env bash
# Cloud Agent / local start script: whisper serve on loopback only.
# Do not bind all interfaces. Host defaults to 127.0.0.1.
set -euo pipefail

cd "$(dirname "$0")/.."

host="${WHISPER_SERVE_HOST:-127.0.0.1}"
port="${WHISPER_SERVE_PORT:-8765}"

exec python3 -m whisper.serve --host "${host}" --port "${port}"
