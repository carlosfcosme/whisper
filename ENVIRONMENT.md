# Environment

Single source of truth for the Cloud Agent and local development environment.
This file states three rules: **install**, **no-weight-pull**, and **localhost-only**.

The environment **forces** `127.0.0.1` and **no default weight fetch**.
`.cursor/verify.sh` checks both; CI job `environment-verify` runs that script.

No model files and no secrets belong in this repository.

## Install

Cloud Agent bootstrap is `.cursor/environment.json`, which runs:

```bash
bash .cursor/install.sh
```

Install sources `.cursor/env.sh`, which exports:

- `WHISPER_BIND_HOST=127.0.0.1`
- `WHISPER_ALLOW_WEIGHT_FETCH=0`

The install script is idempotent. It:

1. Installs `ffmpeg` if it is missing (required for audio decoding).
2. Installs a CPU build of PyTorch (`torch==2.5.1+cpu`) so the CLI and tests run without a GPU.
3. Installs this checkout in editable mode with the `dev` extras (`pytest`, `black`, `isort`, `flake8`, `scipy`).

It does **not** call `load_model` and does **not** download checkpoints.

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

Full environment check (what CI runs):

```bash
bash .cursor/verify.sh
```

## No-weight-pull

Default weight fetch is **off**. `whisper._download` refuses remote/WAN URLs
(including `openaipublic.azureedge.net` and the CLI default model `turbo`)
unless `WHISPER_ALLOW_WEIGHT_FETCH=1` is set explicitly.

Do not add a precache or `load_model(...)` step to `.cursor/install.sh`.
Do not commit `.pt` files or any other weight artifacts.

Cache hits stay local. The cache directory is `$XDG_CACHE_HOME/whisper`
when `XDG_CACHE_HOME` is set, otherwise `~/.cache/whisper`.

`tests/test_transcribe.py` calls `load_model` for official models and is a
weight pull unless those files are already cached. The CI matrix job that
runs `tiny` / `tiny.en` must set `WHISPER_ALLOW_WEIGHT_FETCH=1`. Verify
and install must not.

## Localhost-only

Bind is forced to **`127.0.0.1`**. `whisper.env_policy.require_bind_127_0_0_1`
rejects `0.0.0.0` and every other address before a socket is opened.

`.cursor/start.sh` runs `python3 -m whisper.serve --host 127.0.0.1`.
That path is a weights-free health endpoint: no `load_model`, no CDN.

`.cursor/environment.json` publishes no ports.

Do not put tokens, passwords, or other secrets in `ENVIRONMENT.md`,
`.cursor/environment.json`, install scripts, or the git tree.
