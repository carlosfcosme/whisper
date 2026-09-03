#!/usr/bin/env bash
# Idempotent Cloud Agent setup: offline default, no Hub, CPU-only, 127.0.0.1.
# Does not pull Whisper model weights. No secrets are written.
set -euo pipefail

cd "$(dirname "$0")/.."

# Encode policy for this environment (sourced by verify / later shells).
export WHISPER_OFFLINE=1
export WHISPER_NO_HUB=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cat > .cursor/whisper-policy.env <<'EOF'
WHISPER_OFFLINE=1
WHISPER_NO_HUB=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
WHISPER_DEVICE=cpu
WHISPER_BIND_HOST=127.0.0.1
EOF

# ffmpeg is required at runtime for audio decoding.
if ! command -v ffmpeg >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends ffmpeg
fi

# CPU-only PyTorch (no CUDA). Entry points land on ~/.local/bin (on PATH).
pip install --break-system-packages \
  "numpy" torch==2.5.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple

pip install --break-system-packages -e ".[dev]"

echo "whisper environment ready (offline, no Hub, CPU-only, 127.0.0.1):"
python3 -c "import whisper; from whisper.policy import DEFAULT_DEVICE, BIND_HOST, offline_enabled, hub_disabled; print('  whisper', whisper.__version__, '| device', DEFAULT_DEVICE, '| bind', BIND_HOST, '| offline', offline_enabled(), '| no_hub', hub_disabled())"
