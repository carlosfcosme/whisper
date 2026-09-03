import os

import pytest

import whisper
from whisper.tokenizer import get_tokenizer


@pytest.mark.requires_weights
@pytest.mark.parametrize("model_name", whisper.available_models())
def test_transcribe(model_name: str):
    checkpoint = whisper.local_checkpoint_path(model_name)
    if checkpoint is None:
        pytest.skip(
            f"offline: weights for {model_name!r} are not on disk "
            "(CI does not download checkpoints)"
        )
    model = whisper.load_model(checkpoint, device="cpu")
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")
    if not os.path.isfile(audio_path):
        pytest.skip("offline: tests/jfk.flac fixture is missing")

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
