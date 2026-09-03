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

Cloud Agent setup pre-caches the `tiny` and `tiny.en` model weights into
`~/.cache/whisper` during install (see `.cursor/install.sh`). After that,
transcription and the `.cursor/verify.sh` self-check run from the local cache
with no network required.

No secrets are stored in this repository.
