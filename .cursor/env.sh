# Forced Cloud Agent / local-dev policy. Sourced by install, start, and verify.
# No secrets. No model-weight URLs. CPU-only offline default.
export WHISPER_BIND_HOST=127.0.0.1
export WHISPER_ALLOW_WEIGHT_FETCH=0
export WHISPER_DEVICE=cpu
export CUDA_VISIBLE_DEVICES=""
