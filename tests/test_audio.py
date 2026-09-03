import numpy as np

from whisper.audio import SAMPLE_RATE, load_audio, log_mel_spectrogram


def test_audio(jfk_audio_path):
    audio = load_audio(jfk_audio_path)
    assert audio.ndim == 1
    assert SAMPLE_RATE * 10 < audio.shape[0] < SAMPLE_RATE * 12
    assert 0 < audio.std() < 1

    mel_from_audio = log_mel_spectrogram(audio)
    mel_from_file = log_mel_spectrogram(jfk_audio_path)

    assert np.allclose(mel_from_audio, mel_from_file)
    assert mel_from_audio.max() - mel_from_audio.min() <= 2.0


def test_tiny_fixture_wav_loads_locally(sample_audio_path):
    assert not sample_audio_path.startswith(("http://", "https://"))
    assert "huggingface" not in sample_audio_path.lower()
    audio = load_audio(sample_audio_path)
    assert audio.ndim == 1
    assert 0 < audio.shape[0] < SAMPLE_RATE * 2
    assert audio.std() > 0
