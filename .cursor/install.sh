#!/usr/bin/env bash
# Idempotent Cloud Agent setup for openai-whisper.
# Installs the ffmpeg system dependency and the package (with dev extras)
# using a CPU build of PyTorch so tests and the `whisper` CLI run without a GPU.
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

# Pre-cache model weights so a fresh agent can transcribe instantly and offline,
# instead of relying on a first-run download (or on test runs happening to fetch
# them). These are the small models used by the CPU test subset and the CLI demo.
# Override the set with WHISPER_PRECACHE_MODELS (space-separated) if needed.
WHISPER_PRECACHE_MODELS="${WHISPER_PRECACHE_MODELS:-tiny.en tiny}"
echo "Pre-caching whisper models: ${WHISPER_PRECACHE_MODELS}"
python3 - "$WHISPER_PRECACHE_MODELS" <<'PY'
import sys

import whisper

for name in sys.argv[1].split():
    # load_model downloads to ~/.cache/whisper and verifies the SHA256, so this
    # is idempotent: an already-cached, valid checkpoint is reused, not refetched.
    whisper.load_model(name)
    print(f"  cached {name}")
PY

echo "whisper environment ready:"
python3 -c "import whisper, torch; print('  whisper', whisper.__version__, '| torch', torch.__version__)"
