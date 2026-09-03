#!/usr/bin/env bash
# End-to-end self-check for the openai-whisper Cloud Agent environment.
# Proves the environment actually works, CPU-only and without downloading any
# weights: asserts the pre-cached weights exist, runs the CPU test subset (the
# same selector CI uses), and transcribes the bundled sample audio with an
# assertion on the expected text. Exits non-zero on any failure.
set -euo pipefail

cd "$(dirname "$0")/.."

# Localhost-only: point all WAN egress at a dead local port so any attempt to
# pull from the internet fails fast. This proves the pre-cached model weights
# and bundled assets are sufficient with no WAN pull (the one-time weight fetch
# happens in install.sh, not here). Loopback traffic is exempt via no_proxy.
export http_proxy="http://127.0.0.1:9"
export https_proxy="http://127.0.0.1:9"
export HTTP_PROXY="http://127.0.0.1:9"
export HTTPS_PROXY="http://127.0.0.1:9"
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="localhost,127.0.0.1,::1"
# CPU-only: hide any CUDA devices so the whole check runs on CPU.
export CUDA_VISIBLE_DEVICES=""
echo "== Localhost-only, CPU-only mode: WAN egress blocked; CUDA hidden =="

echo "== Pre-cache assertion (must not download weights) =="
# The models used below (CLI: tiny.en; test subset: tiny + tiny.en) must already
# be cached by install.sh. Fail fast if missing rather than attempting a download.
cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/whisper"
missing=0
for weight in tiny.en.pt tiny.pt; do
  if [ -f "$cache_dir/$weight" ]; then
    echo "  found $cache_dir/$weight"
  else
    echo "  MISSING $cache_dir/$weight"
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  echo "FAIL: model weights are not pre-cached; verify.sh must not download them. Run .cursor/install.sh first."
  exit 1
fi

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
whisper tests/jfk.flac --model tiny.en --language en --device cpu \
  --output_dir "$out_dir" --output_format txt >/dev/null 2>&1
transcript="$(cat "$out_dir/jfk.txt")"
echo "  transcript: ${transcript}"
if ! grep -qi "my fellow Americans" "$out_dir/jfk.txt"; then
  echo "FAIL: expected phrase 'my fellow Americans' not found in transcription"
  exit 1
fi
rm -rf "$out_dir"

echo "== VERIFY OK: environment is provably working =="
