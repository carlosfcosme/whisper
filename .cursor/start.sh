#!/usr/bin/env bash
# Localhost-only health server. Host is hardcoded to 127.0.0.1.
# Does not load model weights.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -m whisper.serve --host 127.0.0.1 --port "${PORT:-8765}"
