"""Fixture paths are local (in-repo). Does not pull weights."""

import importlib.util
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK = os.path.join(ROOT, "scripts", "check_no_remote_fixtures.py")


def _load_fixtures():
    path = os.path.join(ROOT, "whisper", "fixtures.py")
    spec = importlib.util.spec_from_file_location("whisper_fixtures_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixtures = _load_fixtures()


def test_guard_rejects_remote_fixture_urls():
    urls = (
        "https://huggingface.co/openai/whisper-tiny/resolve/main/jfk.flac",
        "http://example.com/sample.wav",
        "https://hf.co/datasets/foo/bar/resolve/main/audio.flac",
    )
    for url in urls:
        assert fixtures.is_remote_fixture_url(url)
        with pytest.raises(fixtures.RemoteFixtureError, match="local"):
            fixtures.assert_local_fixture(url, must_exist=False)


def test_sample_audio_is_in_repo_file(sample_audio_path):
    assert os.path.isfile(sample_audio_path)
    assert os.path.basename(sample_audio_path) == "jfk.flac"
    assert os.path.abspath(sample_audio_path) == os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "jfk.flac"
    )
    assert not sample_audio_path.lower().startswith(("http://", "https://"))
    assert "huggingface" not in sample_audio_path.lower()
    assert os.path.getsize(sample_audio_path) > 0


def test_tiny_local_fixture_exists(tiny_audio_path):
    assert os.path.isfile(tiny_audio_path)
    assert os.path.basename(tiny_audio_path) == "tiny.wav"
    assert os.path.abspath(tiny_audio_path) == os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tiny.wav"
    )
    assert not tiny_audio_path.lower().startswith(("http://", "https://"))
    assert "huggingface" not in tiny_audio_path.lower()
    assert os.path.getsize(tiny_audio_path) > 0


def test_write_tiny_wav_refuses_remote_target(tmp_path):
    with pytest.raises(fixtures.RemoteFixtureError):
        fixtures.write_tiny_wav(
            "https://huggingface.co/foo/tiny.wav",
        )
    written = fixtures.write_tiny_wav(os.path.join(str(tmp_path), "generated.wav"))
    assert os.path.isfile(written)
    assert not written.lower().startswith(("http://", "https://"))


def test_in_repo_assets_are_local(tiktoken_asset_path, mel_filters_path):
    for path in (tiktoken_asset_path, mel_filters_path):
        assert os.path.isfile(path)
        assert not path.lower().startswith(("http://", "https://"))
        assert "huggingface" not in path.lower()


def test_conftest_declares_no_remote_urls():
    text = open(os.path.join(ROOT, "tests", "conftest.py"), encoding="utf-8").read()
    lowered = text.lower()
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "huggingface" not in lowered
    assert "hf.co" not in lowered


def test_ci_script_accepts_local_fixtures():
    result = subprocess.run(
        [sys.executable, CHECK],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK: fixtures are local" in result.stdout


def test_ci_script_fails_on_remote_conftest(tmp_path):
    tests_dir = tmp_path / "tests"
    github = tmp_path / ".github" / "workflows"
    whisper = tmp_path / "whisper"
    tests_dir.mkdir()
    github.mkdir(parents=True)
    whisper.mkdir()
    (tests_dir / "conftest.py").write_text(
        'SAMPLE = "https://huggingface.co/openai/whisper-tiny/resolve/main/jfk.flac"\n',
        encoding="utf-8",
    )
    (tests_dir / "jfk.flac").write_bytes(b"x")
    (tests_dir / "tiny.wav").write_bytes(b"x")
    (whisper / "fixtures.py").write_text("# local guard\n", encoding="utf-8")
    (github / "test.yml").write_text(
        "pytest -k 'not test_transcribe'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "add", "-f", "tests/jfk.flac", "tests/tiny.wav", "tests/conftest.py"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    env = os.environ.copy()
    env["WHISPER_FIXTURE_ROOT"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, CHECK],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode != 0
    assert "remote fixture" in result.stderr.lower()


def test_offline_env_is_set():
    assert os.environ.get("WHISPER_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
