# Contributing

The [README Setup](README.md#setup) path (`pip install openai-whisper`) installs the
package for inference. It does **not** install the test/lint extras or pin the
CPU PyTorch build that CI uses. Those commands live in
[`.github/workflows/test.yml`](.github/workflows/test.yml) and
[`.cursor/install.sh`](.cursor/install.sh); this file copies them for a clean
machine.

No API keys or other secrets are required. The default test command below does
not download model weights.

## Requirements

- Python 3.8–3.13 (the CI matrix; see `.github/workflows/test.yml`)
- [`ffmpeg`](https://ffmpeg.org/) on `PATH` (the runtime audio decoder)
- A CPU is enough. Tests marked `requires_cuda` are skipped on CPU.

## Clean-machine install

Two install paths are actually used. A `python3 -m venv` is **not** one of
them (and fails on stock Debian/Ubuntu until `python3-venv` is installed).

### Cloud Agent / system Python (`.cursor/install.sh`)

This is the path a fresh Ubuntu machine without conda uses. It matches the
Python 3.12 / 3.13 CI rows (`numpy` unpinned, `torch==2.5.1+cpu`).

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends ffmpeg

pip install --break-system-packages \
  "numpy" torch==2.5.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple

pip install --break-system-packages -e ".[dev]"
```

Dev extras (`pyproject.toml` `[project.optional-dependencies] dev`): `black`,
`flake8`, `isort`, `pytest`, `scipy`. Entry points land on `~/.local/bin`.

### GitHub Actions (conda)

```bash
conda install -n test ffmpeg python=3.12
# with that env on PATH:
pip3 install .["dev"] "numpy" torch==2.5.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple
```

Older CI rows pin `numpy<2` and older `torch==<version>+cpu`. The full matrix
is in `.github/workflows/test.yml`. CI installs the package non-editable
(`.["dev"]`), not `-e`.

## Tests (no weights)

`tests/test_transcribe.py` is the only test that calls `whisper.load_model` and
fetches checkpoints. Skip it on a clean machine:

```bash
pytest --durations=0 -vv -k 'not test_transcribe' -m 'not requires_cuda'
```

This runs the audio (`tests/jfk.flac`, needs `ffmpeg`), tokenizer, CPU timing,
and normalizer tests (17 passed, 22 deselected on the current suite).

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
pip install --upgrade pre-commit
pre-commit install --install-hooks
pre-commit run --all-files
```

Hooks and versions are in [`.pre-commit-config.yaml`](.pre-commit-config.yaml)
(`black`, `isort --profile black`, `flake8`, plus the stock file checks). CI
runs this as its own job before the pytest matrix.
