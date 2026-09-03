# Make and tox

There is no Make or tox workflow in this repository. `make test` and `tox` are
not project commands.

## Inventory

| Artifact | Present |
|----------|---------|
| `Makefile`, `makefile`, `GNUmakefile` | No |
| `tox.ini`, `tox.toml` | No |
| `[tool.tox]` in [`pyproject.toml`](pyproject.toml) | No |
| CI step that runs `make` or `tox` | No |

A host may still have a `make` binary. Without a Makefile it has no project
targets (`No rule to make target 'test'`). `tox` is not a dependency of
`openai-whisper` or of the `dev` extra.

Do not add a Makefile or tox config in a drive-by change.

## What to use instead

Install and test with pip and pytest. That is what
[`.github/workflows/test.yml`](.github/workflows/test.yml) and
[`.cursor/install.sh`](.cursor/install.sh) already do.

```bash
# system decoder
sudo apt-get install -y --no-install-recommends ffmpeg

# CPU PyTorch (Cloud Agent / CI shape)
pip install "numpy" torch==2.5.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple

pip install -e ".[dev]"
```

The `dev` extra is `black`, `flake8`, `isort`, `pytest`, and `scipy`.

## Tests without model weights

[`tests/test_transcribe.py`](tests/test_transcribe.py) calls
`whisper.load_model()`, which fetches checkpoints from
`openaipublic.azureedge.net` into `$XDG_CACHE_HOME/whisper` or
`~/.cache/whisper`. Skip that module unless you mean to download weights.
Do not commit `.pt` files.

Weight-free default (audio, tokenizer, CPU timing, normalizer):

```bash
pytest --durations=0 -vv -k 'not test_transcribe' -m 'not requires_cuda'
```

CI also selects `test_transcribe[tiny]` and `test_transcribe[tiny.en]`, which
**do** download those two checkpoints:

```bash
pytest --durations=0 -vv \
  -k 'not test_transcribe or test_transcribe[tiny] or test_transcribe[tiny.en]' \
  -m 'not requires_cuda'
```

`@pytest.mark.requires_cuda` tests in [`tests/test_timing.py`](tests/test_timing.py)
are skipped on CPU.

## Lint

Lint is pre-commit, not make or tox. See
[`.pre-commit-config.yaml`](.pre-commit-config.yaml):

```bash
pip install --upgrade pre-commit
pre-commit install --install-hooks
pre-commit run --all-files
```

## Secrets

Install and the weight-free suite need no credentials. Publishing uses a
GitHub Actions secret (`PYPI_API_TOKEN`); do not put tokens or keys in the
tree.
