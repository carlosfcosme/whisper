"""Fixture paths must be local files. CI fails on http(s) / Hugging Face fixture URLs."""

import os
import subprocess
import sys

import pytest

from tests.local_fixtures import (
    SAMPLE_AUDIO_PATH,
    TESTS_DIR,
    assert_local_fixture_path,
    remote_fixture_url_hits,
)
from tests.local_fixtures import sample_audio_path as resolved_sample_audio
from tests.local_fixtures import write_tiny_wav


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/jfk.flac",
        "https://huggingface.co/datasets/foo/resolve/main/jfk.flac",
        "https://cdn-lfs.huggingface.co/repos/xx/jfk.flac",
        "hf://datasets/foo/jfk.flac",
        "huggingface.co/openai/whisper-tiny/resolve/main/audio.wav",
        "hf.co/datasets/foo/clip.wav",
    ],
)
def test_assert_local_fixture_path_rejects_remote_urls(url):
    with pytest.raises(ValueError, match="local fixture"):
        assert_local_fixture_path(url)


def test_assert_local_fixture_path_rejects_missing_file(tmp_path):
    missing = tmp_path / "no-such-fixture.wav"
    with pytest.raises(FileNotFoundError):
        assert_local_fixture_path(missing)


def test_committed_sample_audio_is_local_file():
    path = resolved_sample_audio()
    assert path == SAMPLE_AUDIO_PATH
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 0
    assert not path.startswith(("http://", "https://", "hf://"))


def test_sample_audio_path_fixture(sample_audio_path):
    assert sample_audio_path == SAMPLE_AUDIO_PATH
    assert os.path.isfile(sample_audio_path)
    assert "huggingface" not in sample_audio_path.lower()
    assert "http" not in sample_audio_path.lower()


def test_tiny_audio_path_is_local_and_loadable(tiny_audio_path):
    assert os.path.isfile(tiny_audio_path)
    assert tiny_audio_path.endswith(".wav")
    assert os.path.getsize(tiny_audio_path) > 44  # WAV header + samples
    from whisper.audio import SAMPLE_RATE, load_audio

    audio = load_audio(tiny_audio_path)
    assert audio.ndim == 1
    # 0.25s at 16 kHz; allow ffmpeg resampling slack.
    assert SAMPLE_RATE * 0.1 < audio.size < SAMPLE_RATE * 0.5


def test_write_tiny_wav_creates_local_file(tmp_path):
    path = write_tiny_wav(tmp_path / "sine.wav", seconds=0.1, sample_rate=8000)
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 44


def test_no_remote_fixture_urls_in_tests_tree():
    hits = remote_fixture_url_hits()
    assert hits == [], "remote fixture URLs:\n" + "\n".join(
        f"{path}:{line}: {text}" for path, line, text in hits
    )


def test_local_fixtures_scanner_exits_zero():
    proc = subprocess.run(
        [sys.executable, os.path.join(TESTS_DIR, "local_fixtures.py")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "no remote fixture URLs" in proc.stdout


def test_scanner_fails_when_https_fixture_url_is_planted():
    planted = TESTS_DIR / "_planted_remote_fixture.py"
    planted.write_text('clip = "https://example.com/clip.wav"\n')
    try:
        hits = remote_fixture_url_hits()
        assert any("https://example.com/clip.wav" in text for _, _, text in hits)
        proc = subprocess.run(
            [sys.executable, os.path.join(TESTS_DIR, "local_fixtures.py")],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 1
        assert "https://example.com/clip.wav" in proc.stderr
    finally:
        planted.unlink()


def test_ci_runs_local_fixture_scanner():
    workflow = (TESTS_DIR.parent / ".github" / "workflows" / "test.yml").read_text()
    assert "local-fixtures:" in workflow
    assert "python3 tests/local_fixtures.py" in workflow


def test_scanner_allows_hub_hostname_blocklist_without_url():
    from tests.local_fixtures import _REMOTE_RE

    assert _REMOTE_RE.search("https://example.com/a.wav")
    assert _REMOTE_RE.search("huggingface.co/openai/whisper-tiny")
    assert _REMOTE_RE.search("hf://datasets/foo/bar")
    assert _REMOTE_RE.search("hf.co/datasets/foo") is not None
    assert _REMOTE_RE.search('"huggingface.co"') is None
    assert _REMOTE_RE.search('"hf.co"') is None
