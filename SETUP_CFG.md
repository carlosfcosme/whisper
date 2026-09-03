# setup.cfg / pyproject.toml test paths

This repository does **not** ship a `setup.cfg`. Setuptools metadata and the
pytest collection path live in [`pyproject.toml`](pyproject.toml).

A `setup.cfg` `[tool:pytest] testpaths = …` section is therefore not part of
this environment. Pytest from the repo root used to find `tests/` only by
convention. That path is now named in `[tool.pytest.ini_options]`.

## Inventory

| Path or section | Present? | Role |
|-----------------|----------|------|
| `setup.cfg` | No | Classic setuptools / `[tool:pytest]` file — absent |
| `setup.py` | No | Historical only (see [CHANGELOG.md](CHANGELOG.md)) |
| `pytest.ini` | No | No standalone pytest config |
| [`pyproject.toml`](pyproject.toml) | Yes | Packaging + pytest `testpaths` |
| `[tool.setuptools]` | Yes | Installable package is `whisper` |
| `[tool.setuptools.packages.find] exclude` | Yes | `tests*` is **not** installed as a package |
| `[tool.pytest.ini_options] testpaths` | Yes | `tests` |

Do not add a `setup.cfg` unless you intend to own that workflow. Prefer
[`pyproject.toml`](pyproject.toml).

## Test paths

`[tool.pytest.ini_options]` in [`pyproject.toml`](pyproject.toml):

```toml
testpaths = [ "tests" ]
```

That is the same path as `pytest tests`. Collection stays inside `tests/`.
The `whisper/` package is excluded from setuptools (`exclude = [ "tests*" ]`)
and is not a pytest root.

| Path | Kind |
|------|------|
| [`tests/conftest.py`](tests/conftest.py) | fixtures / `requires_cuda` marker |
| [`tests/test_audio.py`](tests/test_audio.py) | audio / ffmpeg |
| [`tests/test_normalizer.py`](tests/test_normalizer.py) | text normalizers |
| [`tests/test_timing.py`](tests/test_timing.py) | DTW / median filter (some CUDA) |
| [`tests/test_tokenizer.py`](tests/test_tokenizer.py) | tokenizer |
| [`tests/test_transcribe.py`](tests/test_transcribe.py) | `load_model` — **downloads weights** |
| [`tests/test_setup_cfg_paths.py`](tests/test_setup_cfg_paths.py) | this inventory (no weights) |
| [`tests/jfk.flac`](tests/jfk.flac) | local sample audio fixture |

## Tests (no weights)

[`tests/test_transcribe.py`](tests/test_transcribe.py) calls
`whisper.load_model()`, which downloads checkpoints from
`openaipublic.azureedge.net` into `$XDG_CACHE_HOME/whisper` or
`~/.cache/whisper`. Skip that module unless you intend to fetch weights.
Do not commit `.pt` files.

Weight-free suite (packaging paths, audio, tokenizer, CPU timing, normalizer):

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

## Secrets

No credentials are required to install the package or run the weight-free
tests. Do not put tokens or credentials in the tree.
