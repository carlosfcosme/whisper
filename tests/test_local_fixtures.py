"""Fixture paths are local (in-repo or tempfile). Does not pull weights."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from whisper.audio import load_audio
from whisper.fixtures import (
    RemoteFixtureError,
    assert_local_fixture,
    is_remote_fixture_url,
    write_tiny_wav,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = REPO_ROOT / ".github" / "scripts" / "check_no_remote_fixtures.py"


def test_guard_rejects_remote_fixture_urls():
    urls = (
        "https://huggingface.co/openai/whisper-tiny/resolve/main/jfk.flac",
        "http://example.com/sample.wav",
        "https://hf.co/datasets/foo/bar/resolve/main/audio.flac",
    )
    for url in urls:
        assert is_remote_fixture_url(url)
        with pytest.raises(RemoteFixtureError, match="local"):
            assert_local_fixture(url, must_exist=False)


def test_sample_audio_is_in_repo_file(sample_audio_path):
    path = Path(sample_audio_path)
    assert path.is_file()
    assert path.name == "jfk.flac"
    assert path.suffix == ".flac"
    assert path.resolve().parent == (REPO_ROOT / "tests").resolve()
    assert not sample_audio_path.lower().startswith(("http://", "https://"))
    assert "huggingface" not in sample_audio_path.lower()
    assert path.stat().st_size > 0


def test_tiny_tempfile_audio_is_local(tiny_audio_path, tmp_path):
    path = Path(tiny_audio_path)
    assert path.is_file()
    assert path.suffix == ".wav"
    assert path.resolve().parent == tmp_path.resolve()
    assert not str(path).lower().startswith(("http://", "https://"))
    audio = load_audio(str(path))
    assert audio.ndim == 1
    assert audio.shape[0] > 0


def test_write_tiny_wav_refuses_remote_target():
    with pytest.raises(RemoteFixtureError):
        write_tiny_wav("https://huggingface.co/foo/tiny.wav")


def test_in_repo_assets_are_local(tiktoken_asset_path, mel_filters_path):
    for path in (tiktoken_asset_path, mel_filters_path):
        assert Path(path).is_file()
        assert not path.lower().startswith(("http://", "https://"))
        assert "huggingface" not in path.lower()


def test_conftest_declares_no_remote_urls():
    text = (REPO_ROOT / "tests" / "conftest.py").read_text().lower()
    assert "http://" not in text
    assert "https://" not in text
    assert "huggingface" not in text
    assert "hf.co" not in text


def test_ci_script_accepts_local_fixtures():
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT)],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "OK: fixtures are local" in result.stdout


def test_offline_env_is_set():
    assert os.environ.get("WHISPER_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
