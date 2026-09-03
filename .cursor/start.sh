#!/usr/bin/env bash
# Bind the weights-free health server to 127.0.0.1 only. No Hub, no checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -m whisper.serve --host 127.0.0.1 --port 8765
