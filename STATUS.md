# Status

Commercial / development environment notes for offline work.

These notes state the intended policy: **commercial offline**,
**localhost-only**, and **no weight pulls**. That is a user policy for
offline work, not a guarantee that an unseeded Cloud Agent already has
cached checkpoints.

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

**No weight pulls** is policy for commercial/dev offline work, not a
property of a fresh install:

- `.cursor/install.sh` on this branch installs ffmpeg, CPU PyTorch, and
  the package. It does not download or cache model checkpoints.
- `whisper` without `--model` defaults to `turbo`. `load_model()` downloads
  any named checkpoint that is not already on disk.
- Do not download model checkpoints during offline work.
- Do not commit model files (`.pt`, `.pth`, `.bin`, `.onnx`) to this
  repository.
- Use weights already on disk. The cache directory is
  `$XDG_CACHE_HOME/whisper` when `XDG_CACHE_HOME` is set, otherwise
  `~/.cache/whisper`.
- For offline work, pass an already-cached model such as `--model tiny`
  or `--model tiny.en`. Do not use the `turbo` default unless that file
  is already cached.

A separate Cloud Agent install change can pre-cache `tiny` / `tiny.en` at
setup time. That is install-time caching only. After those files exist
locally, runtime must follow no weight pulls.

## Localhost-only

This environment is **localhost-only**. Whisper is a local CLI and Python
library. Do not expose it as a network service. If a notebook, Gradio app,
or helper HTTP server is started for development, bind it to `127.0.0.1`
only.

## Secrets and artifacts

No secrets belong in this repository. No model files belong in git.
