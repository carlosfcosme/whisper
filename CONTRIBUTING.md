# Contributing

This is the clone-to-test path for a **clean local machine**.

- **No secrets.** No API keys, tokens, or credentials are required.
- **No weights.** Default tests do not call `whisper.load_model` and do not
  download checkpoints.
- **Localhost only.** Whisper is a local CLI and library. Helpers that listen
  must use `whisper.bind.listen_loopback` / `bind_loopback` (`127.0.0.1` only).
  `0.0.0.0` and any non-loopback host are refused. CI job `loopback-listen`
  fails on a non-loopback LISTEN.

The [README Setup](README.md#setup) `pip install openai-whisper` path is for
inference only. It does not install the test extras.

## Requirements

- git
- Python 3.8–3.13 (the CI matrix in `.github/workflows/test.yml`)
- [`ffmpeg`](https://ffmpeg.org/) on `PATH` (runtime audio decoder)
- `python3-venv` on Debian/Ubuntu (`python3 -m venv` needs it)

A CPU is enough. Tests marked `requires_cuda` are skipped without a GPU.

## Clone and install

```bash
git clone https://github.com/carlosfcosme/whisper.git
cd whisper

# Debian/Ubuntu. Other OS ffmpeg commands are in README.md.
sudo apt-get update
sudo apt-get install -y --no-install-recommends ffmpeg python3-venv

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# CPU wheel matching the Python 3.12 / 3.13 rows in CI.
python -m pip install "numpy" torch==2.5.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple

# Editable checkout plus dev extras: black, flake8, isort, pytest, scipy.
python -m pip install -e ".[dev]"
```

You may need [Rust](https://www.rust-lang.org/learn/get-started) if tiktoken
has no wheel for your platform; see README Setup.

## Tests (no weights)

`tests/test_transcribe.py` is the only test that calls `whisper.load_model` and
fetches checkpoints from `openaipublic.azureedge.net`. Skip it:

```bash
pytest --durations=0 -vv -k 'not test_transcribe' -m 'not requires_cuda'
```

This runs the audio (`tests/jfk.flac`, needs `ffmpeg`), tokenizer, CPU timing,
and normalizer tests. It does not write under `$XDG_CACHE_HOME/whisper` or
`~/.cache/whisper`.

Do not smoke-test with the `whisper` CLI. It defaults to `--model turbo` and
will download that checkpoint. GitHub Actions additionally runs
`test_transcribe[tiny]` and `test_transcribe[tiny.en]`; those fetch weights
and are outside this bootstrap.

## Lint

```bash
python -m pip install pre-commit
pre-commit run --all-files
```

Hooks and versions are in [`.pre-commit-config.yaml`](.pre-commit-config.yaml).
