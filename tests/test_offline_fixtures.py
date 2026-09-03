import os


def test_sample_audio_fixture_is_local(sample_audio_path):
    expected = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jfk.flac")
    assert os.path.isfile(sample_audio_path)
    assert os.path.basename(sample_audio_path) == "jfk.flac"
    assert sample_audio_path == expected
    assert os.path.isabs(sample_audio_path)
    assert not sample_audio_path.startswith(("http://", "https://"))


def test_tiny_wav_fixture_is_local(tiny_audio_path):
    expected = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny.wav")
    assert os.path.isfile(tiny_audio_path)
    assert os.path.basename(tiny_audio_path) == "tiny.wav"
    assert tiny_audio_path == expected
    assert not tiny_audio_path.startswith(("http://", "https://"))


def test_packaged_assets_are_local():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets = (
        os.path.join(root, "whisper", "assets", "mel_filters.npz"),
        os.path.join(root, "whisper", "assets", "gpt2.tiktoken"),
        os.path.join(root, "whisper", "assets", "multilingual.tiktoken"),
        os.path.join(root, "whisper", "normalizers", "english.json"),
    )
    for path in assets:
        assert os.path.isfile(path), path
        assert not path.startswith(("http://", "https://"))
