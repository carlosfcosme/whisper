#!/usr/bin/env bash
# Idempotent Cloud Agent setup for whisper (sovereign).
# Installs ffmpeg and the package (with dev extras) using a CPU build of
# PyTorch so tests and the `whisper` CLI run without a GPU.
# Do not download or commit model checkpoints (.pt). Install is deps only.
# Do not install or call Hugging Face Hub / huggingface_hub / from_pretrained.
set -euo pipefail

# Offline for any transitive Hub client. Whisper itself does not use the Hub.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "$(dirname "$0")/.."

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
python3 -c "import whisper, torch; print('  whisper', whisper.__version__, '| torch', torch.__version__)"
