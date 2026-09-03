#!/usr/bin/env bash
# Idempotent Cloud Agent setup for openai-whisper.
# Installs the ffmpeg system dependency and the package (with dev extras)
# using a CPU build of PyTorch so tests and the `whisper` CLI run without a GPU.
#
# Sovereign Cloud Agent path defaults (also set by tests/conftest.py and CI):
#   WHISPER_CPU_ONLY=1            → whisper.default_device() == cpu
#   WHISPER_NO_WEIGHT_DOWNLOAD=1  → refuse checkpoint auto-download (incl. HF Hub)
#   WHISPER_LOCALHOST_ONLY=1      → helper listeners bind 127.0.0.1
set -euo pipefail

cd "$(dirname "$0")/.."

export WHISPER_CPU_ONLY="${WHISPER_CPU_ONLY:-1}"
export WHISPER_NO_WEIGHT_DOWNLOAD="${WHISPER_NO_WEIGHT_DOWNLOAD:-1}"
export WHISPER_LOCALHOST_ONLY="${WHISPER_LOCALHOST_ONLY:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

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
python3 -c "import whisper, torch; print('  whisper', whisper.__version__, '| torch', torch.__version__, '| device', whisper.default_device(), '| bind', whisper.default_bind_host())"
