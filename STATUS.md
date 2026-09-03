# Status

This repository is **sovereign** and **commercial**.

- **CI fails if weights are committed.** Job `no-committed-weights` runs
  `scripts/check_committed_artifacts.sh` and exits non-zero on tracked
  checkpoints (`.pt`, `.onnx`, …) or large binaries.
- **CPU default.** `whisper.DEFAULT_DEVICE` and the CLI `--device` default
  are `cpu`.
- **Serve binds 127.0.0.1.** `whisper.localhost.serve_bind_host` /
  `whisper.serve` bind loopback only and reject `0.0.0.0`.

## Commercial

Whisper's code and model weights are released under the [MIT License](LICENSE).
Commercial use is permitted, including using, copying, modifying, merging,
publishing, distributing, sublicensing, and selling copies, subject to the
license conditions (retain the copyright and permission notice; the software
is provided "as is", without warranty).

This is a plain informational note only. It does not modify the license or
grant any additional rights.

## Localhost-only, no weight pulls

`.cursor/install.sh` installs ffmpeg, CPU PyTorch, and the package with dev
extras. It does not call `whisper.load_model()` and does not download
checkpoints.

`.cursor/verify.sh` runs the artifact guard, tool/import checks, and the CPU
tests that do not load models. It skips `test_transcribe` and the `whisper`
CLI so a missing cache cannot trigger a weight pull.

No remote inference service is started on a public interface. No secrets are
stored in this repository.
