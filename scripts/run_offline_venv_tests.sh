#!/usr/bin/env bash
# Create an isolated venv and run offline tests (no Hub, loopback-only).
# Uses --system-site-packages + pip --no-index --no-deps so the venv install
# does not hit the network. Model/WAN fetches are refused at runtime and
# blocked by a 127.0.0.1:9 proxy.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKDIR="${OFFLINE_VENV_DIR:-$(mktemp -d)}"
PYTHON="${OFFLINE_PYTHON:-python3}"

"$PYTHON" -m venv --system-site-packages "$WORKDIR/venv"
VENV_PY="$WORKDIR/venv/bin/python"

"$VENV_PY" -m pip install --no-index --no-deps -e "$ROOT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export http_proxy="http://127.0.0.1:9"
export https_proxy="http://127.0.0.1:9"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"

exec "$VENV_PY" -m pytest -vv \
  tests/test_offline_fetch.py \
  tests/test_loopback_bind.py \
  tests/test_gitignore_weights.py \
  "$@"
