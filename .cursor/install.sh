#!/usr/bin/env bash
# Idempotent Cloud Agent setup for openai-whisper.
# Installs the ffmpeg system dependency and the package (with dev extras)
# using a CPU build of PyTorch so tests and the `whisper` CLI run without a GPU.
set -euo pipefail

cd "$(dirname "$0")/.."

export WHISPER_CPU_ONLY="${WHISPER_CPU_ONLY:-1}"
export WHISPER_NO_WEIGHT_DOWNLOAD="${WHISPER_NO_WEIGHT_DOWNLOAD:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

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

echo "whisper environment ready:"
python3 -c "import whisper, torch; print('  whisper', whisper.__version__, '| torch', torch.__version__, '| device', whisper.default_device(), '| bind', whisper.require_loopback_host())"
