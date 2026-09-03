#!/usr/bin/env bash
# Lint + offline/bind guards. No torch, no Hub, no model download.
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=""
export WHISPER_OFFLINE="1"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export HF_HUB_DISABLE_TELEMETRY="1"

echo "== lint (black / isort / flake8) =="
black --check .
isort --check-only --profile black -l 88 --trailing-comma --multi-line 3 .
flake8 --max-line-length 88 --ignore E203,E501,W503,W504 whisper tests scripts

echo "== no committed weights =="
python3 scripts/check_no_weights.py

echo "== 127.0.0.1-only bind =="
python3 scripts/check_loopback_bind.py

echo "== no Hub / model download =="
python3 scripts/check_no_download.py

echo "OK: lint-test CI (offline, loopback-only)"
