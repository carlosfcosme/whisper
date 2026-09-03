# Test fixtures

Sample audio and other fixtures live in this checkout. Tests resolve them
from the source tree. **Do not download sample audio or model weights** to
run the default suite.

The sample-audio path is not an env var. Enforcement lives in
`whisper/offline.py` (local fixtures, Hub refused, bind `127.0.0.1`, CPU
default). `python3 whisper/offline.py --check` fails CI if weights are
committed. No secrets are required.

## Sample audio

| Path | Role |
|------|------|
| [`tests/jfk.flac`](jfk.flac) | Bundled JFK excerpt used by the audio and transcribe tests |

The path used to be implicit: each test did
`os.path.join(os.path.dirname(__file__), "jfk.flac")`. That is now the named
fixture `sample_audio_path` in [`conftest.py`](conftest.py), which always
points at the file next to this README.

Do **not** fetch a JFK clip (or any other speech sample) from the network.
`sample_audio_path` is a local filesystem path, never an `http://` or
`https://` URL.

### Who uses it

- [`test_audio.py`](test_audio.py) — `load_audio` / `log_mel_spectrogram`
  only. No `load_model`. **No weight pull.**
- [`test_transcribe.py`](test_transcribe.py) — the same local file plus
  `whisper.load_model(...)`. That **does** fetch checkpoints on a cache
  miss. Skip it to stay offline:

  ```bash
  pytest -k 'not test_transcribe' -m 'not requires_cuda'
  ```

## Other in-repo fixtures (no network pull)

These ship with the package. Paths are also implicit
(`os.path.dirname(__file__)`), not env vars.

| Path | Used by |
|------|---------|
| `whisper/assets/mel_filters.npz` | `whisper.audio.mel_filters` |
| `whisper/assets/gpt2.tiktoken` | English tokenizer |
| `whisper/assets/multilingual.tiktoken` | Multilingual tokenizer |
| `whisper/normalizers/english.json` | `EnglishTextNormalizer` |

`data/meanwhile.json` is paper-evaluation metadata, not a pytest fixture.

## Not fixtures

Model checkpoints (`.pt`) are **not** in the repository. `load_model` writes
them to `$XDG_CACHE_HOME/whisper` or `~/.cache/whisper` and pulls from
`openaipublic.azureedge.net` on a miss. Do not commit weights. Do not add
a download step for sample audio or weights in Cloud Agent setup.
