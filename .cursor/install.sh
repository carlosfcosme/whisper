#!/usr/bin/env bash
# Sovereign Cloud Agent setup for openai-whisper.
#
# Agent path (deps already present, e.g. environment snapshot):
#   no WAN, no weight pull, no bind.
# First-time bootstrap may apt/pip packages only — never checkpoints.
set -euo pipefail

cd "$(dirname "$0")/.."

deps_ready() {
  command -v ffmpeg >/dev/null 2>&1 || return 1
  python3 - <<'PY'
import importlib
import sys

for name in (
    "whisper",
    "torch",
    "numpy",
    "pytest",
    "black",
    "isort",
    "flake8",
    "scipy",
):
    try:
        importlib.import_module(name)
    except ImportError:
        sys.exit(1)
PY
}

# Never call whisper.load_model or fetch openaipublic.azureedge.net.
# This script binds no sockets (127.0.0.1 only if a helper is added later).

if deps_ready; then
  echo "sovereign: deps present — no WAN, no weight pull, localhost bind only"
else
  echo "bootstrap: installing packages (no weight pull)"

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

echo "whisper environment ready:"
python3 -c "import whisper, torch; print('  whisper', whisper.__version__, '| torch', torch.__version__)"
