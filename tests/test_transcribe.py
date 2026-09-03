import os
from typing import Optional

import pytest

import whisper
from whisper.tokenizer import get_tokenizer


def _cached_model_path(model_name: str) -> Optional[str]:
    url = whisper._MODELS[model_name]
    default = os.path.join(os.path.expanduser("~"), ".cache")
    root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")
    target = os.path.join(root, os.path.basename(url))
    return target if os.path.isfile(target) else None


@pytest.mark.requires_local_weights
@pytest.mark.parametrize("model_name", whisper.available_models())
def test_transcribe(model_name: str, sample_audio_path):
    if _cached_model_path(model_name) is None:
        pytest.skip(
            "offline fixtures only; tests must not download weights or contact the Hub"
        )

    model = whisper.load_model(model_name, device=whisper.DEFAULT_DEVICE)

    language = "en" if model_name.endswith(".en") else None
    result = model.transcribe(
        sample_audio_path, language=language, temperature=0.0, word_timestamps=True
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
