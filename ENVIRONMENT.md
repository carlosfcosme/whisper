# Environment

Single source of truth for the Cloud Agent and local development environment.
This file states three rules: **install**, **no-weight-pull**, and **localhost-only**.

No model files and no secrets belong in this repository.

## Install

Cloud Agent bootstrap is `.cursor/environment.json`, which runs:

```bash
bash .cursor/install.sh
```

The install script is idempotent. It:

1. Installs `ffmpeg` if it is missing (required for audio decoding).
2. Installs a CPU build of PyTorch (`torch==2.5.1+cpu`) so the CLI and tests run without a GPU.
3. Installs this checkout in editable mode with the `dev` extras (`pytest`, `black`, `isort`, `flake8`, `scipy`).

Entry points (`whisper`, `pytest`, linters) land on the active interpreter's
scripts directory (`~/.local/bin` on this Cloud Agent image, or a virtualenv
`bin` when one is active).

To confirm the package imported without fetching checkpoints:

```bash
python3 -c "import whisper, torch; print('whisper', whisper.__version__, '| torch', torch.__version__)"
```

No-weight-pull tests (no `load_model`, so no checkpoint download):

```bash
pytest -k 'not test_transcribe' -m 'not requires_cuda'
```

## No-weight-pull

Install does **not** download Whisper checkpoints. Do not add a precache or
`load_model(...)` step to `.cursor/install.sh`. Do not commit `.pt` files or
any other weight artifacts.

`whisper.load_model()` and the CLI (default model `turbo`) fetch from
`openaipublic.azureedge.net` on a cache miss. That is a weight pull. Do not
run those paths during environment setup.

If a checkpoint is already on disk, the cache is
`$XDG_CACHE_HOME/whisper` when `XDG_CACHE_HOME` is set, otherwise
`~/.cache/whisper`. Cache hits stay local; cache misses are a network pull.

`tests/test_transcribe.py` calls `load_model` for every official model and
is therefore a weight pull unless those files are already cached. Skip it
for no-weight-pull work.

## Localhost-only

This environment is local-only. Whisper is a CLI; there is no application
server and `.cursor/environment.json` publishes no ports.

Do not bind listeners on `0.0.0.0` or any public interface. If a helper
process must accept connections (demo, notebook, proxy), bind `127.0.0.1`
only.

Do not put tokens, passwords, or other secrets in `ENVIRONMENT.md`,
`.cursor/environment.json`, install scripts, or the git tree.
