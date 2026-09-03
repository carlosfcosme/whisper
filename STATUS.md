# Status

This repository is **commercial**. This Cloud Agent environment is
**localhost-only** and does **not** pull model weights.

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

`.cursor/verify.sh` runs tool/import checks and the CPU tests that do not
load models. It skips `test_transcribe` and the `whisper` CLI so a missing
cache cannot trigger a weight pull. The CLI defaults to `turbo`; invoking it
without a local checkpoint would attempt a download and is out of scope here.

Any HTTP serve path binds **127.0.0.1** only (`whisper.localhost.serve_bind_host`;
`0.0.0.0` is rejected). `load_model` and the CLI default to **CPU**.

CI job `no-committed-weights` runs `scripts/check_committed_artifacts.sh` and
fails if model weights (`.pt`, `.onnx`, …) or large binaries are committed.

No remote inference service is started on a public interface. No secrets are
stored in this repository.
