# Status

Commercial / development environment notes for offline work.

This environment is **commercial offline**, **localhost-only**, and performs
**no weight pulls**.

## Commercial offline

Whisper's code and model weights are released under the [MIT License](LICENSE).
Commercial use is permitted subject to the license conditions (retain the
copyright and permission notice; the software is provided "as is", without
warranty).

This note does not modify the license or grant additional rights. See the
[model card](model-card.md) for recommended-use guidance.

**Commercial offline** means local inference only: use the library and
checkpoints already on this machine. Do not call remote transcription APIs
and do not fetch weights while working offline.

## No weight pulls

The commercial/dev environment performs **no weight pulls**:

- Do not download model checkpoints during offline work.
- Do not commit model files (`.pt`, `.pth`, `.bin`, `.onnx`) to this
  repository.
- Use weights already on disk. The cache directory is
  `$XDG_CACHE_HOME/whisper` when `XDG_CACHE_HOME` is set, otherwise
  `~/.cache/whisper`.
- The `whisper` CLI defaults to `turbo`. If that checkpoint is not already
  cached, the CLI will attempt a download. For offline work, pass an
  already-cached model such as `--model tiny` or `--model tiny.en`.

A separate Cloud Agent install change can pre-cache `tiny` / `tiny.en` at
setup time. That is install-time caching only. After those files exist
locally, runtime must perform no weight pulls.

## Localhost-only

This environment is **localhost-only**. Whisper is a local CLI and Python
library. Do not expose it as a network service. If a notebook, Gradio app,
or helper HTTP server is started for development, bind it to `127.0.0.1`
only.

## Secrets and artifacts

No secrets belong in this repository. No model files belong in git.
