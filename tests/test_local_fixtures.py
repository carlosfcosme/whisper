import subprocess
from pathlib import Path

import pytest

import whisper.audio as audio_mod
from whisper.audio import SAMPLE_RATE, load_audio, log_mel_spectrogram
from whisper.local_fixtures import (
    REGISTERED,
    assert_local_path,
    check_registered_fixtures,
    is_remote_fixture_url,
    resolve,
)

REMOTE_FIXTURE_URLS = (
    "https://huggingface.co/datasets/x/resolve/main/a.wav",
    "http://example.com/clip.wav",
    "https://hf.co/openai/whisper-tiny/resolve/main/tiny.pt",
    "hf://openai/whisper-tiny/tone.wav",
    "s3://bucket/fixtures/tone.wav",
    "huggingface://openai/whisper-tiny",
)


@pytest.mark.parametrize("url", REMOTE_FIXTURE_URLS)
def test_remote_urls_are_detected(url):
    assert is_remote_fixture_url(url)


@pytest.mark.parametrize("url", REMOTE_FIXTURE_URLS)
def test_assert_local_path_rejects_remote(url):
    with pytest.raises(ValueError, match="Remote fixture"):
        assert_local_path(url)
    with pytest.raises(ValueError, match="Remote fixture"):
        resolve(url)


@pytest.mark.parametrize("url", REMOTE_FIXTURE_URLS)
def test_load_audio_rejects_remote_without_ffmpeg(url, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("ffmpeg must not run for remote fixture URLs")

    monkeypatch.setattr(audio_mod, "run", boom)
    with pytest.raises(ValueError, match="Remote fixture"):
        load_audio(url)


def test_registered_fixtures_are_local_files():
    assert check_registered_fixtures() == 0
    for name in REGISTERED:
        path = resolve(name)
        assert path.is_file()
        assert not is_remote_fixture_url(str(path))
        assert not str(path).startswith(("http://", "https://", "hf://"))


def test_tiny_wav_is_local_and_loadable(tiny_wav_path):
    path = Path(tiny_wav_path)
    assert path.is_file()
    assert path.suffix == ".wav"
    assert path.stat().st_size < 8192
    assert not tiny_wav_path.startswith(("http://", "https://"))
    audio = load_audio(tiny_wav_path)
    assert audio.ndim == 1
    assert 0 < audio.std() < 1
    assert SAMPLE_RATE * 0.05 < audio.shape[0] < SAMPLE_RATE * 0.2
    mel = log_mel_spectrogram(tiny_wav_path)
    assert mel.shape[0] == 80


def test_tiny_raw_bytes_are_local(tiny_pcm_path):
    path = Path(tiny_pcm_path)
    data = path.read_bytes()
    assert path.suffix == ".raw"
    assert len(data) == 3200
    assert path.stat().st_size < 8192
    assert data[:4] != b"RIFF"
    assert b"http://" not in data
    assert b"huggingface" not in data.lower()


def test_jfk_flac_is_registered_local(sample_audio_path):
    path = Path(sample_audio_path)
    assert path.is_file()
    assert path.name == "jfk.flac"
    assert not is_remote_fixture_url(sample_audio_path)


def test_resolve_rejects_path_escape(tmp_path):
    inner = tmp_path / "inner"
    inner.mkdir()
    (tmp_path / "outside.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="escapes"):
        resolve("../outside.bin", root=inner)


def test_check_fails_when_registry_has_remote_url(monkeypatch, capsys):
    monkeypatch.setitem(
        REGISTERED, "evil", "https://huggingface.co/datasets/x/resolve/main/a.wav"
    )
    assert check_registered_fixtures() == 1
    captured = capsys.readouterr()
    assert "remote registry" in captured.err


def test_ci_script_passes_on_local_fixtures():
    root = Path(__file__).resolve().parents[1]
    script = root / ".github" / "scripts" / "fail-remote-fixture-urls.sh"
    subprocess.check_call(["bash", str(script)], cwd=str(root))


def test_no_field_brain_or_keys_in_fixtures():
    root = Path(__file__).resolve().parent / "fixtures"
    banned = (b"field-brain", b"field_brain", b"api_key", b"sk-", b"BEGIN ")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        data = path.read_bytes().lower()
        for token in banned:
            assert token.lower() not in data, path
