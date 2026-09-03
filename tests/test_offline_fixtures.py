from pathlib import Path

import pytest

from whisper.audio import load_audio

_PATHS_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_paths():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "whisper_test_fixture_paths", _PATHS_DIR / "paths.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


paths = _load_paths()


def test_tiny_wav_is_committed_local_file(sample_audio_path):
    path = Path(sample_audio_path)
    assert path.is_file()
    assert path.name == "tiny.wav"
    assert path.parent.name == "fixtures"
    assert path.stat().st_size > 0
    assert path.stat().st_size < 64 * 1024
    assert path.is_absolute()
    assert not sample_audio_path.startswith(("http://", "https://"))
    assert "huggingface" not in sample_audio_path.lower()


def test_jfk_flac_is_in_repo_local_file(jfk_audio_path):
    path = Path(jfk_audio_path)
    assert path.is_file()
    assert path.name == "jfk.flac"
    assert not jfk_audio_path.startswith(("http://", "https://"))


def test_fixture_path_rejects_remote_urls():
    remote = "https://" + "huggingface.co" + "/openai/whisper-tiny/tiny.wav"
    with pytest.raises(ValueError, match="local"):
        paths.local_path(remote)
    with pytest.raises(ValueError, match="local"):
        paths.fixture_path(remote)


def test_is_remote_url_detects_wan_and_allows_loopback():
    assert paths.is_remote_url("https://" + "example.invalid/audio.wav")
    assert paths.is_remote_url("http://" + "huggingface.co" + "/x")
    assert not paths.is_remote_url("/tmp/local.wav")
    assert not paths.is_remote_url("http://127.0.0.1:8765/health")


def test_tempfile_wav_is_local_and_loadable(tmp_path):
    dest = tmp_path / "generated.wav"
    written = paths.write_sine_wav(dest, seconds=0.1)
    assert written.is_file()
    assert not str(written).startswith(("http://", "https://"))
    audio = load_audio(str(written))
    assert audio.ndim == 1
    assert audio.shape[0] > 0


def test_write_sine_wav_refuses_remote_dest():
    with pytest.raises(ValueError, match="remote"):
        paths.write_sine_wav("https://" + "example.invalid/out.wav")
