# Contributing
Install with `pip install -e ".[dev]"` (ffmpeg on PATH; no API keys or other secrets).
Test with `pytest --durations=0 -vv -k 'not test_transcribe' -m 'not requires_cuda'` so `whisper.load_model` does not fetch weights.
