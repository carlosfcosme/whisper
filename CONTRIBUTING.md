# Contributing

The [README Setup](README.md#setup) path (`pip install openai-whisper`) installs the
package for inference. It does **not** install the test/lint extras or pin the
CPU PyTorch build that CI uses. Those commands live in
[`.github/workflows/test.yml`](.github/workflows/test.yml) and
[`.cursor/install.sh`](.cursor/install.sh); this file writes them down for a
clean machine.

No API keys or other secrets are required. The default test command below does
not download model weights.

## Requirements

- Python 3.8–3.13 (the CI matrix; see `.github/workflows/test.yml`)
- [`ffmpeg`](https://ffmpeg.org/) on `PATH` (the runtime audio decoder)
- A CPU is enough. Tests marked `requires_cuda` are skipped on CPU.

## Clean-machine install

Use a virtualenv. `--break-system-packages` appears only in
`.cursor/install.sh`, which installs into the Cloud Agent system interpreter.

```bash
# Debian/Ubuntu. Other OS ffmpeg commands are in README.md.
sudo apt-get update
sudo apt-get install -y --no-install-recommends ffmpeg

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

GitHub Actions uses conda instead of apt + venv, then a non-editable install:

```bash
conda install -n test ffmpeg python=3.12
# with that env on PATH:
pip3 install .["dev"] "numpy" torch==2.5.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple
```

Older CI rows pin `numpy<2` and older `torch==<version>+cpu`. The full matrix is
in `.github/workflows/test.yml`.

## Tests (no weights)

`tests/test_transcribe.py` is the only test that calls `whisper.load_model` and
fetches checkpoints. Skip it on a clean machine:

```bash
pytest --durations=0 -vv -k 'not test_transcribe' -m 'not requires_cuda'
```

This runs the audio (`tests/jfk.flac`, needs `ffmpeg`), tokenizer, CPU timing,
and normalizer tests.

CI also exercises the two smallest transcribe cases. That **does** download the
`tiny` and `tiny.en` checkpoints into `$XDG_CACHE_HOME/whisper` or
`~/.cache/whisper`:

```bash
pytest --durations=0 -vv \
  -k 'not test_transcribe or test_transcribe[tiny] or test_transcribe[tiny.en]' \
  -m 'not requires_cuda'
```

The `whisper` CLI defaults to `--model turbo` and will download that checkpoint.
Do not use the CLI to smoke-test a weight-free install.

## Lint

```bash
python -m pip install pre-commit
pre-commit install --install-hooks
pre-commit run --all-files
```

Hooks and versions are in [`.pre-commit-config.yaml`](.pre-commit-config.yaml)
(`black`, `isort --profile black`, `flake8`, plus the stock file checks).
