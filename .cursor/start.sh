#!/usr/bin/env bash
# Bind the weights-free health server to 127.0.0.1 only.
# Do not pass an all-interface host.
set -euo pipefail

cd "$(dirname "$0")/.."

host="${WHISPER_SERVE_HOST:-127.0.0.1}"
port="${WHISPER_SERVE_PORT:-8765}"

exec python3 -m whisper.serve --host "${host}" --port "${port}"
