# Make and tox

This repository does **not** ship a Make- or tox-driven developer environment.
The install, test, and lint commands live in GitHub Actions and the Cloud Agent
setup script instead.

## Inventory

| Tool | Present? | Where this was checked |
|------|----------|------------------------|
| `Makefile`, `makefile`, `GNUmakefile` | No | repository root and `git ls-files` |
| `tox.ini`, `tox.toml` | No | repository root and `git ls-files` |
| `[tool.tox]` | No | [`pyproject.toml`](pyproject.toml) |
| CI invoking `make` or `tox` | No | [`.github/workflows/test.yml`](.github/workflows/test.yml), [`.github/workflows/python-publish.yml`](.github/workflows/python-publish.yml) |

A system `make` binary may exist on the host. Without a Makefile, `make test`
(or any other target) is not a project command. `tox` is not a project
dependency.

Do not add a Makefile or `tox.ini` unless you intend to own that workflow.

## Implicit environment

Dev extras from [`pyproject.toml`](pyproject.toml): `black`, `flake8`, `isort`,
`pytest`, `scipy`. Runtime also needs [`ffmpeg`](https://ffmpeg.org/) to decode
audio.

Cloud Agent install ([`.cursor/install.sh`](.cursor/install.sh)):

```bash
# ffmpeg if missing
sudo apt-get install -y --no-install-recommends ffmpeg

pip install --break-system-packages \
  "numpy" torch==2.5.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple

pip install --break-system-packages -e ".[dev]"
```

GitHub Actions ([`.github/workflows/test.yml`](.github/workflows/test.yml))
installs `ffmpeg` and Python with conda, then:

```bash
pip3 install .["dev"] <numpy-requirement> torch==<matrix>+cpu \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple
```

There is no tox env list and no `make test` target.

## Tests (no weights)

[`tests/test_transcribe.py`](tests/test_transcribe.py) calls
`whisper.load_model()`, which downloads checkpoints from
`openaipublic.azureedge.net` into `$XDG_CACHE_HOME/whisper` or
`~/.cache/whisper`. Skip that module unless you intend to fetch weights.
Do not commit `.pt` files.

Weight-free suite (audio, tokenizer, CPU timing, normalizer):

```bash
pytest --durations=0 -vv -k 'not test_transcribe' -m 'not requires_cuda'
```

CI additionally runs `test_transcribe[tiny]` and `test_transcribe[tiny.en]`,
which **do** download those two checkpoints:

```bash
pytest --durations=0 -vv \
  -k 'not test_transcribe or test_transcribe[tiny] or test_transcribe[tiny.en]' \
  -m 'not requires_cuda'
```

CUDA tests (`@pytest.mark.requires_cuda` in [`tests/test_timing.py`](tests/test_timing.py))
are skipped on CPU.

## Lint

CI uses [`.pre-commit-config.yaml`](.pre-commit-config.yaml), not make or tox:

```bash
pip install --upgrade pre-commit
pre-commit install --install-hooks
pre-commit run --all-files
```

## Secrets

No credentials are required to install the package or run the weight-free
tests. Release publishing reads a GitHub Actions secret; do not put tokens
or credentials in the tree.
