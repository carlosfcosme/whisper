#!/usr/bin/env bash
# Weights-free health server. Bind is forced to 127.0.0.1.
# Does not call load_model and does not fetch checkpoints.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=env.sh
source .cursor/env.sh

if [[ "${WHISPER_BIND_HOST}" != "127.0.0.1" ]]; then
  echo "FAIL: WHISPER_BIND_HOST must be 127.0.0.1, got ${WHISPER_BIND_HOST}" >&2
  exit 1
fi

exec python3 -m whisper.serve \
  --host 127.0.0.1 \
  --port "${WHISPER_SERVE_PORT:-8765}"
