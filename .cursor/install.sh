#!/usr/bin/env bash
# Idempotent Cloud Agent setup for openai-whisper.
# pip only (no uv). CPU PyTorch. No model-weight download.
# This script starts no server; start.sh binds 127.0.0.1 only.
# WHISPER_OFFLINE_INSTALL=1 skips apt/pip for network-disabled tests.
set -euo pipefail

cd "$(dirname "$0")/.."

export WHISPER_NO_WEIGHT_DOWNLOAD="${WHISPER_NO_WEIGHT_DOWNLOAD:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

if [[ "${WHISPER_OFFLINE_INSTALL:-0}" != "1" ]]; then
  # ffmpeg is required at runtime for audio decoding.
  if ! command -v ffmpeg >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends ffmpeg
  fi

  # CPU-only PyTorch (no CUDA in the Cloud Agent VM), pinned to the version CI
  # uses for Python 3.12. --break-system-packages installs into the user site so
  # the `whisper`, `pytest`, and lint entrypoints land on ~/.local/bin (on PATH).
  pip install --break-system-packages \
    "numpy" torch==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

  # Editable install of the package plus dev tooling (pytest, black, isort, flake8, scipy).
  pip install --break-system-packages -e ".[dev]"
fi

# Import-only readiness check. Isolated cache so install cannot leave checkpoints.
# Do not call load_model or the whisper CLI — those fetch checkpoints.
_install_cache="$(mktemp -d)"
trap 'rm -rf "${_install_cache}"' EXIT
echo "whisper environment ready:"
XDG_CACHE_HOME="${_install_cache}" \
  WHISPER_NO_WEIGHT_DOWNLOAD=1 \
  HF_HUB_OFFLINE=1 \
  python3 -c \
  "import whisper, torch; print('  whisper', whisper.__version__, '| torch', torch.__version__, '| device', whisper.DEFAULT_DEVICE)"

if find "${_install_cache}" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.safetensors' -o -name 'pytorch_model.bin' \) | grep -q .; then
  echo "install.sh must not download model weights" >&2
  exit 1
fi
