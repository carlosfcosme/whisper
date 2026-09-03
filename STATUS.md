# Status

This repository is **sovereign**.

- **CI fails if weights are committed.** Job `no-committed-weights` runs
  `scripts/check_no_weights.py` and exits non-zero on tracked checkpoints
  (`.pt`, `.onnx`, `.safetensors`, …), compiled binaries, or files over 10 MiB.
- **CPU default.** `whisper.DEFAULT_DEVICE` and the CLI `--device` default
  are `cpu`. CUDA is opt-in (`--device cuda`).
- **Serve binds 127.0.0.1.** `whisper.serve` / `whisper-serve` bind loopback
  only and reject `0.0.0.0`.

## Localhost-only, no weight pulls

Tests set `CUDA_VISIBLE_DEVICES=""` and Hugging Face Hub offline flags.
`test_transcribe` skips unless a checkpoint is already on disk so CI does
not download weights.

No remote inference service is started on a public interface. No secrets are
stored in this repository.
