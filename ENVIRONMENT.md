# Environment

Local development for this Whisper checkout is **localhost-only**.
Install must not pull model weights. Do not commit secrets.

This file is the source of truth for those three rules. Usage examples
live in [README.md](README.md). Cloud Agent bootstrap is
[`.cursor/environment.json`](.cursor/environment.json) →
[`bash .cursor/install.sh`](.cursor/install.sh).

## Localhost

Whisper is a local CLI and Python library. There is no application
server and `environment.json` publishes no ports.

If a helper process must listen (notebook, demo, proxy), bind
`127.0.0.1` only. Do not bind `0.0.0.0` or any public interface.

```bash
# Import-only check (no checkpoint download):
python3 -c "import whisper, torch; print(whisper.__version__, torch.__version__)"
```

## No weights

`.cursor/install.sh` installs `ffmpeg`, CPU PyTorch (`torch==2.5.1+cpu`),
and an editable `.[dev]` install. It does **not** call `load_model` and
must not grow a precache step.

Do not commit `.pt` files or any other checkpoint. First-use downloads
go to `$XDG_CACHE_HOME/whisper` when set, otherwise `~/.cache/whisper`.
A cache miss hits `openaipublic.azureedge.net`. That is a weight pull.

`whisper.load_model()`, the `whisper` CLI (default model `turbo`), and
`tests/test_transcribe.py` all fetch on a cache miss. Skip those paths
when staying weight-free.

```bash
# Tests that do not call load_model:
pytest -k 'not test_transcribe' -m 'not requires_cuda'
```

## No secrets

Whisper needs no API keys, tokens, or passwords. Do not put secrets in
`ENVIRONMENT.md`, `.cursor/environment.json`, install scripts, tests,
or the git tree. Use the host secret store if a personal workflow needs
credentials; never commit them.
