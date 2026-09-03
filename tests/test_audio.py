import os.path

import numpy as np

from whisper.audio import SAMPLE_RATE, load_audio, log_mel_spectrogram


def test_sample_audio_fixture_is_local(sample_audio_path):
    from whisper.offline import is_hub_url, require_local_path

    assert os.path.isfile(sample_audio_path)
    assert os.path.basename(sample_audio_path) == "jfk.flac"
    assert not sample_audio_path.startswith(("http://", "https://", "hf://"))
    assert not is_hub_url(sample_audio_path)
    assert require_local_path(sample_audio_path) == os.path.abspath(sample_audio_path)


def test_audio(sample_audio_path):
    audio = load_audio(sample_audio_path)
    assert audio.ndim == 1
    assert SAMPLE_RATE * 10 < audio.shape[0] < SAMPLE_RATE * 12
    assert 0 < audio.std() < 1

    mel_from_audio = log_mel_spectrogram(audio)
    mel_from_file = log_mel_spectrogram(sample_audio_path)

    assert np.allclose(mel_from_audio, mel_from_file)
    assert mel_from_audio.max() - mel_from_audio.min() <= 2.0
