#!/usr/bin/env bash
# Idempotent Cloud Agent setup for openai-whisper.
# Installs the ffmpeg system dependency and the package (with dev extras)
# using a CPU build of PyTorch so tests and the `whisper` CLI run without a GPU.
#
# pip / uv install must not pull model weights. This script uses pip
# (`uv pip install .` is the equivalent and is also weight-free).
# Do not call load_model or the whisper CLI here.
set -euo pipefail

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

# Import-only readiness check. Isolated cache so install cannot leave checkpoints.
_install_cache="$(mktemp -d)"
trap 'rm -rf "${_install_cache}"' EXIT
echo "whisper environment ready:"
XDG_CACHE_HOME="${_install_cache}" python3 -c \
  "import whisper, torch; print('  whisper', whisper.__version__, '| torch', torch.__version__, '| device', whisper.DEFAULT_DEVICE)"

if find "${_install_cache}" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.safetensors' \) | grep -q .; then
  echo "pip/uv install must not download model weights" >&2
  exit 1
fi
