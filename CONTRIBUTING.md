# Contributing

Clone-to-test is three lines. No API keys or other secrets are required.
`ffmpeg` must be on `PATH` (see [README Setup](README.md#setup)).

```bash
git clone https://github.com/carlosfcosme/whisper.git
cd whisper && pip install -e ".[dev]"
pytest --durations=0 -vv -k 'not test_transcribe' -m 'not requires_cuda'
```

`test_transcribe` is skipped so `whisper.load_model` does not fetch checkpoints.
The remaining tests use the in-repo fixture `tests/jfk.flac`.
