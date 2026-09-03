#!/usr/bin/env bash
# Localhost-only self-check for the openai-whisper Cloud Agent environment.
# Does not pull model weights: skips test_transcribe and the whisper CLI
# (both call load_model, which downloads a checkpoint if it is missing).
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== Tool + import check =="
command -v ffmpeg >/dev/null || { echo "FAIL: ffmpeg not found on PATH"; exit 1; }
command -v whisper >/dev/null || { echo "FAIL: whisper CLI not found on PATH"; exit 1; }
python3 -c "import whisper, torch, numpy; print('  whisper', whisper.__version__, '| torch', torch.__version__, '| numpy', numpy.__version__)"

echo "== CPU tests without model-weight pulls =="
# Exclude test_transcribe: it calls whisper.load_model() for every official
# checkpoint name. Tokenizer / timing / normalizer / audio tests stay local.
pytest -q \
  -k 'not test_transcribe' \
  -m 'not requires_cuda'

echo "== VERIFY OK: localhost-only, no weight pulls =="
