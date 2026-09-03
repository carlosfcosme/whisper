"""Default is offline: Hub/Azure fetch is blocked; tests fail if attempted."""

import hashlib
import importlib.util
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

import whisper
from whisper.offline import (
    ALLOW_DOWNLOADS_ENV,
    downloads_allowed,
    downloads_forbidden,
    offline_forced,
)

from .conftest import DownloadAttempted

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    path = REPO_ROOT / "scripts" / "{}.py".format(name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _clear_download_env(monkeypatch):
    monkeypatch.delenv(ALLOW_DOWNLOADS_ENV, raising=False)
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)


def test_downloads_disallowed_by_default(monkeypatch):
    _clear_download_env(monkeypatch)
    assert downloads_allowed() is False
    assert downloads_forbidden() is True
    assert whisper.downloads_forbidden() is True
    assert offline_forced() is False


def test_offline_env_forces_deny_even_with_opt_in(monkeypatch):
    monkeypatch.setenv(ALLOW_DOWNLOADS_ENV, "1")
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    assert downloads_allowed() is False
    assert offline_forced() is True


def test_opt_in_allows_download_when_not_offline(monkeypatch):
    _clear_download_env(monkeypatch)
    monkeypatch.setenv(ALLOW_DOWNLOADS_ENV, "1")
    assert downloads_allowed() is True
    assert downloads_forbidden() is False


def test_download_refuses_by_default_without_calling_urlopen(tmp_path, monkeypatch):
    _clear_download_env(monkeypatch)
    attempted = []

    def _boom(url, *args, **kwargs):
        attempted.append(url)
        raise AssertionError("urlopen must not run: {}".format(url))

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    fake_url = (
        "https://example.invalid/"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/"
        "tiny.pt"
    )
    with pytest.raises(RuntimeError, match="offline"):
        whisper._download(fake_url, str(tmp_path), False)
    assert attempted == []


def test_hub_urlopen_fails_the_test():
    with pytest.raises(DownloadAttempted, match="forbidden"):
        urllib.request.urlopen("https://huggingface.co")


def test_azure_weight_urlopen_fails_the_test():
    with pytest.raises(DownloadAttempted, match="forbidden"):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
        )


def test_zero_addr_is_not_a_loopback_bind():
    """Localhost-only constraint: all-interfaces is never an allowed bind."""
    assert "0.0.0.0" != "127.0.0.1"


def test_download_uses_valid_local_cache_without_network(tmp_path, monkeypatch):
    content = b"cached-weights"
    digest = hashlib.sha256(content).hexdigest()
    url = "https://example.invalid/{}/tiny.pt".format(digest)
    (tmp_path / "tiny.pt").write_bytes(content)
    attempted = []

    def _boom(url, *args, **kwargs):
        attempted.append(url)
        raise AssertionError("urlopen must not run for a valid cache")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    result = whisper._download(url, str(tmp_path), False)
    assert result == str(tmp_path / "tiny.pt")
    assert attempted == []


def test_mismatched_cache_does_not_redownload_when_offline(tmp_path, monkeypatch):
    _clear_download_env(monkeypatch)
    (tmp_path / "tiny.pt").write_bytes(b"wrong")
    fake_url = (
        "https://example.invalid/"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/"
        "tiny.pt"
    )
    attempted = []

    def _boom(url, *args, **kwargs):
        attempted.append(url)
        raise AssertionError("urlopen must not re-download")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    with pytest.raises(RuntimeError, match="offline"):
        whisper._download(fake_url, str(tmp_path), False)
    assert attempted == []


def test_cli_help_does_not_download():
    result = subprocess.run(
        [sys.executable, "-m", "whisper", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--allow_downloads" in result.stdout


def test_check_offline_default_script_passes():
    script = REPO_ROOT / "scripts" / "check_offline_default.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "offline" in result.stdout


def test_assert_no_weight_cache_passes():
    script = REPO_ROOT / "scripts" / "assert_no_weight_cache.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK:" in result.stdout


def test_assert_no_weight_cache_flags_planted_checkpoint(tmp_path):
    cache = _load_script("assert_no_weight_cache")
    planted = tmp_path / "whisper"
    planted.mkdir()
    (planted / "tiny.pt").write_bytes(b"not-weights")
    found = cache.find_cached_weights([planted])
    assert found == [planted / "tiny.pt"]


def test_check_offline_default_flags_tiny_ci_filter(tmp_path):
    gate = _load_script("check_offline_default")
    workflow = tmp_path / ".github" / "workflows"
    workflow.mkdir(parents=True)
    (workflow / "test.yml").write_text(
        "jobs:\n  whisper-test:\n    steps:\n"
        "      - run: pytest -k 'test_transcribe[tiny]'\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    errors = gate.check_workflow_offline(tmp_path)
    assert any("downloaded" in item for item in errors)


def test_huggingface_hub_is_not_a_runtime_dependency():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "huggingface" not in text.lower()
