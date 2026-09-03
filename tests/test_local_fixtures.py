"""Fixture paths are in-repo or temp only. No WAN, no Hub, no keys."""

import subprocess
import sys
from pathlib import Path

import pytest

from whisper.audio import SAMPLE_RATE, load_audio
from whisper.fixtures import (
    IN_REPO_SAMPLE_AUDIO,
    RemoteFixtureError,
    is_remote_fixture_url,
    require_local_fixture,
    tiny_wav_bytes,
    write_tiny_wav,
)

REMOTE_WAV = "https://example.invalid/fixtures/sample.wav"
HUB_FLAC = "https://huggingface.co/datasets/example/resolve/main/jfk.flac"


def test_sample_audio_fixture_is_local(sample_audio_path):
    expected = Path(__file__).resolve().parent / "jfk.flac"
    assert Path(sample_audio_path).is_file()
    assert Path(sample_audio_path).name == "jfk.flac"
    assert Path(sample_audio_path).resolve() == expected
    assert not sample_audio_path.startswith(("http://", "https://"))
    assert "huggingface" not in sample_audio_path.lower()


def test_in_repo_sample_is_tracked_local_file():
    path = require_local_fixture(IN_REPO_SAMPLE_AUDIO)
    assert Path(path).is_file()
    assert path.endswith("tests/jfk.flac") or path.endswith("tests\\jfk.flac")


def test_tiny_wav_fixture_is_temp_local(tiny_wav_path, tmp_path):
    path = Path(tiny_wav_path)
    assert path.is_file()
    assert path.suffix == ".wav"
    assert path.parent == tmp_path
    assert not str(path).startswith(("http://", "https://"))
    audio = load_audio(tiny_wav_path)
    assert audio.ndim == 1
    assert audio.shape[0] == 160
    assert audio.dtype.kind == "f"


def test_tiny_audio_bytes_are_local_wav(tiny_audio_bytes, tmp_path):
    assert tiny_audio_bytes[:4] == b"RIFF"
    assert b"WAVE" in tiny_audio_bytes[:16]
    dest = tmp_path / "from_bytes.wav"
    dest.write_bytes(tiny_audio_bytes)
    loaded = load_audio(str(dest))
    assert loaded.shape[0] == 160
    assert SAMPLE_RATE == 16000


def test_tiny_wav_bytes_helper_matches_writer(tmp_path):
    payload = tiny_wav_bytes()
    written = Path(write_tiny_wav(tmp_path / "copy.wav")).read_bytes()
    assert payload == written
    assert not is_remote_fixture_url(str(tmp_path / "copy.wav"))


def test_require_local_fixture_refuses_remote_urls():
    for url in (REMOTE_WAV, HUB_FLAC, "http://127.0.0.1/clip.wav"):
        assert is_remote_fixture_url(url)
        with pytest.raises(RemoteFixtureError, match="remote"):
            require_local_fixture(url)


def test_load_audio_refuses_remote_fixture_url():
    with pytest.raises(ValueError, match="remote fixture"):
        load_audio(REMOTE_WAV)
    with pytest.raises(ValueError, match="remote fixture"):
        load_audio(HUB_FLAC)


def test_urlopen_hook_blocks_remote_audio_fixture():
    import urllib.request

    with pytest.raises(RuntimeError, match="Hub/weight"):
        urllib.request.urlopen(REMOTE_WAV)


def test_no_field_brain_and_no_keys_in_fixture_module():
    root = Path(__file__).resolve().parents[1]
    text = (root / "whisper" / "fixtures.py").read_text()
    assert "Field-Brain" not in text
    assert "FIELD_BRAIN" not in text
    assert "API_KEY" not in text
    assert "SECRET" not in text


def test_ci_script_accepts_local_fixtures():
    script = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "scripts"
        / "fail-remote-fixture-urls.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok:" in completed.stdout
