import os
from pathlib import Path

import pytest

import whisper
from whisper.tokenizer import get_tokenizer


def _checkpoint_on_disk(model_name: str) -> bool:
    """True when a named checkpoint is already cached. Never downloads."""
    default = os.path.join(os.path.expanduser("~"), ".cache")
    root = Path(os.getenv("XDG_CACHE_HOME", default)) / "whisper"
    url = whisper._MODELS.get(model_name)
    if url is None:
        return os.path.isfile(model_name)
    return (root / os.path.basename(url)).is_file()


@pytest.mark.parametrize("model_name", whisper.available_models())
def test_transcribe(model_name: str):
    if not _checkpoint_on_disk(model_name):
        pytest.skip("no local checkpoint; tests must not download Hub or model weights")

    model = whisper.load_model(model_name, device=whisper.DEFAULT_DEVICE)
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")

    language = "en" if model_name.endswith(".en") else None
    result = model.transcribe(
        audio_path, language=language, temperature=0.0, word_timestamps=True
    )
    assert result["language"] == "en"
    assert result["text"] == "".join([s["text"] for s in result["segments"]])

    transcription = result["text"].lower()
    assert "my fellow americans" in transcription
    assert "your country" in transcription
    assert "do for you" in transcription

    tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages)
    all_tokens = [t for s in result["segments"] for t in s["tokens"]]
    assert tokenizer.decode(all_tokens) == result["text"]
    assert tokenizer.decode_with_timestamps(all_tokens).startswith("<|0.00|>")

    timing_checked = False
    for segment in result["segments"]:
        for timing in segment["words"]:
            assert timing["start"] < timing["end"]
            if timing["word"].strip(" ,") == "Americans":
                assert timing["start"] <= 1.8
                assert timing["end"] >= 1.8
                timing_checked = True

    assert timing_checked
