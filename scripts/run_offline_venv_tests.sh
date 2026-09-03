#!/usr/bin/env bash
# Create an isolated venv and run offline tests (no Hub, loopback-only).
# Prefers `venv` + `pip install --no-index --no-deps`. If ensurepip is
# missing, falls back to `--without-pip` and PYTHONPATH so the suite still
# runs offline. WAN fetches are refused at runtime and blocked by a
# 127.0.0.1:9 proxy.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
WORKDIR="${OFFLINE_VENV_DIR:-$(mktemp -d)}"
PYTHON="${OFFLINE_PYTHON:-python3}"

if "$PYTHON" -m venv --system-site-packages "$WORKDIR/venv"; then
  VENV_PY="$WORKDIR/venv/bin/python"
  "$VENV_PY" -m pip install --no-index --no-deps --no-build-isolation -e "$ROOT"
else
  echo "ensurepip unavailable; using --without-pip venv + PYTHONPATH" >&2
  "$PYTHON" -m venv --without-pip --system-site-packages "$WORKDIR/venv"
  VENV_PY="$WORKDIR/venv/bin/python"
  export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
fi

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
