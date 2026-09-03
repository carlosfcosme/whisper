#!/usr/bin/env bash
# End-to-end self-check for the openai-whisper Cloud Agent environment.
# Proves the environment actually works: runs the CPU test subset (the same
# selector CI uses) and a real transcription of the bundled sample audio with
# an assertion on the expected text. Exits non-zero on any failure.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== Tool + import check =="
command -v ffmpeg >/dev/null || { echo "FAIL: ffmpeg not found on PATH"; exit 1; }
command -v whisper >/dev/null || { echo "FAIL: whisper CLI not found on PATH"; exit 1; }
python3 -c "import whisper, torch, numpy; print('  whisper', whisper.__version__, '| torch', torch.__version__, '| numpy', numpy.__version__)"

echo "== CPU test subset (mirrors CI) =="
# Same selector as .github/workflows/test.yml: skip heavy models, keep tiny/tiny.en.
pytest -q \
  -k 'not test_transcribe or test_transcribe[tiny] or test_transcribe[tiny.en]' \
  -m 'not requires_cuda'

echo "== End-to-end CLI transcription =="
out_dir="$(mktemp -d)"
whisper tests/jfk.flac --model tiny.en --language en \
  --output_dir "$out_dir" --output_format txt >/dev/null 2>&1
transcript="$(cat "$out_dir/jfk.txt")"
echo "  transcript: ${transcript}"
if ! grep -qi "my fellow Americans" "$out_dir/jfk.txt"; then
  echo "FAIL: expected phrase 'my fellow Americans' not found in transcription"
  exit 1
fi
rm -rf "$out_dir"

echo "== VERIFY OK: environment is provably working =="
