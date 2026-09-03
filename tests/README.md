# Test fixtures

Fixture paths live in this checkout. Tests resolve them from the source tree.
**No WAN** is required for sample audio or packaged assets. **No model weights**
and **no secrets** are required for the default suite.

There is no environment variable for the sample-audio path.

## Sample audio

| Path | Role |
|------|------|
| [`tests/jfk.flac`](jfk.flac) | Bundled JFK excerpt used by the audio and transcribe tests |

The path used to be implicit: each test rebuilt
`os.path.join(os.path.dirname(__file__), "jfk.flac")`. That is now the named
fixture `sample_audio_path` in [`conftest.py`](conftest.py), which always
points at the file next to this README.

`sample_audio_path` is a local filesystem path. It is never an `http://` or
`https://` URL. Do not download a JFK clip or any other speech sample.

### Who uses it

- [`test_audio.py`](test_audio.py) — `load_audio` / `log_mel_spectrogram`
  only. No `load_model`. **No weight pull. No WAN.**
- [`test_transcribe.py`](test_transcribe.py) — the same local file plus
  `whisper.load_model(...)`. Checkpoints are **not** fixtures. On a cache
  miss that call hits `openaipublic.azureedge.net`. Skip it to stay offline:

  ```bash
  pytest -k 'not test_transcribe' -m 'not requires_cuda'
  ```

## Other in-repo fixtures (no WAN)

These ship with the package. Paths are resolved from `__file__`, not env vars.

| Path | Used by |
|------|---------|
| `whisper/assets/mel_filters.npz` | `whisper.audio.mel_filters` |
| `whisper/assets/gpt2.tiktoken` | English tokenizer |
| `whisper/assets/multilingual.tiktoken` | Multilingual tokenizer |
| `whisper/normalizers/english.json` | `EnglishTextNormalizer` |

`data/meanwhile.json` is paper-evaluation metadata, not a pytest fixture.

## Not fixtures

Model checkpoints (`.pt`) are not in the repository. `load_model` writes them
to `$XDG_CACHE_HOME/whisper` or `~/.cache/whisper`. Do not commit weights.
Do not add a download step for sample audio or weights.
