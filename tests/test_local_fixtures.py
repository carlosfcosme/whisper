"""Local fixtures exercise load/audio without downloading weights."""

from pathlib import Path

import torch

import whisper
from whisper.audio import SAMPLE_RATE, load_audio, log_mel_spectrogram
from whisper.model import Whisper


def test_toy_checkpoint_loads_offline(toy_checkpoint, toy_dims):
    assert toy_checkpoint.is_file()
    assert toy_checkpoint.suffix == ".pt"
    loaded = whisper.load_model(str(toy_checkpoint))
    assert isinstance(loaded, Whisper)
    assert loaded.dims.n_mels == toy_dims.n_mels
    assert loaded.dims.n_audio_layer == toy_dims.n_audio_layer
    assert loaded.device.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_toy_checkpoint_is_not_in_the_repo(toy_checkpoint):
    repo = Path(__file__).resolve().parents[1]
    assert repo not in toy_checkpoint.resolve().parents
    assert toy_checkpoint.resolve() != repo / toy_checkpoint.name


def test_jfk_audio_fixture_is_local(jfk_audio_path):
    audio = load_audio(str(jfk_audio_path))
    assert audio.ndim == 1
    assert SAMPLE_RATE * 10 < audio.shape[0] < SAMPLE_RATE * 12
    mel = log_mel_spectrogram(audio)
    assert mel.ndim == 2
    assert mel.shape[0] == 80


def test_load_model_named_tiny_does_not_use_fixture_download(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    try:
        whisper.load_model("tiny", download_root=str(tmp_path))
        raised = False
    except RuntimeError as exc:
        raised = True
        assert "offline" in str(exc).lower() or "Hub" in str(exc)
    assert raised
    assert list(tmp_path.rglob("*.pt")) == []
