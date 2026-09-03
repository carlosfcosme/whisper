import os

from tests.conftest import SAMPLE_AUDIO_PATH


def test_sample_audio_fixture_is_local(sample_audio_path):
    assert os.path.isfile(sample_audio_path)
    assert os.path.basename(sample_audio_path) == "jfk.flac"
    assert sample_audio_path == SAMPLE_AUDIO_PATH
    assert os.path.isabs(sample_audio_path)
    assert not sample_audio_path.startswith(("http://", "https://"))


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
