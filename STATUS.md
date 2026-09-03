# Status

This repository is **commercial** and **offline-safe**.

## Commercial

Whisper's code and model weights are released under the [MIT License](LICENSE).
Commercial use is permitted, including using, copying, modifying, merging,
publishing, distributing, sublicensing, and selling copies, subject to the
license conditions (retain the copyright and permission notice; the software
is provided "as is", without warranty).

This is a plain informational note only. It does not modify the license or
grant any additional rights.

## Offline-safe

Cloud Agent setup pre-caches `tiny` and `tiny.en` during install (see
`.cursor/install.sh`; override with `WHISPER_PRECACHE_MODELS`). After that,
those models and `.cursor/verify.sh` run from the local cache with no
network required.

The cache directory is `$XDG_CACHE_HOME/whisper` when `XDG_CACHE_HOME` is
set, otherwise `~/.cache/whisper`.

Offline use applies only to the pre-cached models. The `whisper` CLI
defaults to `turbo`, which is not pre-cached and will attempt a download
unless `--model tiny` or `--model tiny.en` is passed.

No secrets are stored in this repository.
