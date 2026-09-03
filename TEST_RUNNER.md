# Test runner

The test runner is **pytest**. There is no `Makefile` target, no `tox` env,
and no `unittest` discovery. Collection is the `tests/` directory (pytest
default). There is no `[tool.pytest.ini_options]` in
[`pyproject.toml`](pyproject.toml); flags are passed on the command line.

`pytest` is listed in the `dev` extra:

```toml
optional-dependencies.dev = [ "black", "flake8", "isort", "pytest", "scipy" ]
```

## Install (no secrets)

No API keys, tokens, or other credentials are required to install or run
tests. `ffmpeg` must be on `PATH` so `tests/test_audio.py` can decode
[`tests/jfk.flac`](tests/jfk.flac) (see [README Setup](README.md#setup)).

```bash
pip install -e ".[dev]"
```

## Run without weights

[`tests/test_transcribe.py`](tests/test_transcribe.py) is the only test that
calls `whisper.load_model()` and downloads checkpoints from
`openaipublic.azureedge.net` into `$XDG_CACHE_HOME/whisper` or
`~/.cache/whisper`. Skip that module. Do not commit `.pt` files.

```bash
pytest --durations=0 -vv -k 'not test_transcribe' -m 'not requires_cuda'
```

That is **17** collected tests (22 deselected) on the current suite:

| Module | Coverage |
|--------|----------|
| [`tests/test_audio.py`](tests/test_audio.py) | `load_audio` / `log_mel_spectrogram` on `tests/jfk.flac` |
| [`tests/test_tokenizer.py`](tests/test_tokenizer.py) | English and multilingual tokenizers |
| [`tests/test_timing.py`](tests/test_timing.py) | CPU DTW and median filter |
| [`tests/test_normalizer.py`](tests/test_normalizer.py) | English text / number / spelling normalizers |

`@pytest.mark.requires_cuda` is registered in
[`tests/conftest.py`](tests/conftest.py) and used by the GPU equivalence
cases in `tests/test_timing.py`. Those are skipped on CPU.

Do not smoke-test with the `whisper` CLI. It defaults to `--model turbo` and
will download that checkpoint.

## CI (downloads tiny weights)

GitHub Actions ([`.github/workflows/test.yml`](.github/workflows/test.yml))
additionally runs `test_transcribe[tiny]` and `test_transcribe[tiny.en]`.
Those two cases **do** fetch checkpoints. That is not the default local
command:

```bash
pytest --durations=0 -vv \
  -k 'not test_transcribe or test_transcribe[tiny] or test_transcribe[tiny.en]' \
  -m 'not requires_cuda'
```
