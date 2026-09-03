"""Installer must not download weights by default. Localhost only. No WAN."""

import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path

import pytest

import whisper
from whisper.localhost import (
    LOCALHOST_ONLY_ENV,
    NO_WEIGHT_DOWNLOAD_ENV,
    WeightPullError,
    hostname_is_localhost,
    localhost_only_enabled,
    no_weight_download_enabled,
    refuse_weight_network_pull,
    url_is_localhost,
)

ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = ROOT / "scripts" / "check_installer_no_weights.py"

CDN_URL = whisper._MODELS["tiny"]
HF_HUB_URL = "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"
WAN_URLS = (
    CDN_URL,
    HF_HUB_URL,
    "https://example.com/tiny.pt",
    "http://8.8.8.8/tiny.pt",
    "http://10.0.0.1/tiny.pt",
    "https://localhost.evil.com/tiny.pt",
)
LOCALHOST_URLS = (
    "http://127.0.0.1/tiny.pt",
    "http://127.0.0.1:8080/tiny.pt",
    "http://localhost/tiny.pt",
    "http://[::1]/tiny.pt",
    "file:///tmp/tiny.pt",
)


def _forbid_network(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("unit tests must not open a network connection")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(urllib.request, "build_opener", boom)


def test_ci_assertion_script_passes():
    result = subprocess.run(
        ["python3", str(CHECK_SCRIPT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "installer-no-weights: ok" in result.stdout


def test_install_sh_defaults_are_no_download_localhost_only():
    text = (ROOT / ".cursor" / "install.sh").read_text(encoding="utf-8")
    assert "export WHISPER_NO_WEIGHT_DOWNLOAD=" in text
    assert "export WHISPER_LOCALHOST_ONLY=" in text
    assert "XDG_CACHE_HOME" in text
    assert "load_model" not in "\n".join(
        line.split("#", 1)[0] for line in text.splitlines()
    )


def test_environment_json_is_localhost_only():
    raw = (ROOT / ".cursor" / "environment.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    assert "ports" not in data
    assert "0.0.0.0" not in raw
    assert data["install"] == "bash .cursor/install.sh"


@pytest.mark.parametrize("host", ["localhost", "LOCALHOST", "127.0.0.1", "::1"])
def test_hostname_is_localhost(host):
    assert hostname_is_localhost(host)


@pytest.mark.parametrize(
    "host",
    [
        None,
        "",
        "openaipublic.azureedge.net",
        "huggingface.co",
        "8.8.8.8",
        "10.0.0.1",
        "localhost.evil.com",
    ],
)
def test_hostname_is_not_localhost(host):
    assert not hostname_is_localhost(host)


@pytest.mark.parametrize("url", LOCALHOST_URLS)
def test_url_is_localhost(url):
    assert url_is_localhost(url)


@pytest.mark.parametrize("url", WAN_URLS)
def test_url_is_remote(url):
    assert not url_is_localhost(url)


def test_env_defaults_off_without_installer(monkeypatch):
    monkeypatch.delenv(NO_WEIGHT_DOWNLOAD_ENV, raising=False)
    monkeypatch.delenv(LOCALHOST_ONLY_ENV, raising=False)
    assert not no_weight_download_enabled()
    assert not localhost_only_enabled()
    refuse_weight_network_pull(CDN_URL)


def test_no_weight_download_refuses_cdn_without_network(monkeypatch, tmp_path):
    monkeypatch.setenv(NO_WEIGHT_DOWNLOAD_ENV, "1")
    monkeypatch.setenv(LOCALHOST_ONLY_ENV, "1")
    _forbid_network(monkeypatch)

    with pytest.raises(WeightPullError, match="Auto-download"):
        whisper._download(CDN_URL, str(tmp_path), in_memory=False)

    with pytest.raises(WeightPullError):
        whisper._download(HF_HUB_URL, str(tmp_path), in_memory=False)

    with pytest.raises(WeightPullError):
        whisper.load_model("tiny", download_root=str(tmp_path))

    assert list(tmp_path.iterdir()) == []


def test_localhost_only_refuses_wan_but_names_loopback(monkeypatch):
    monkeypatch.delenv(NO_WEIGHT_DOWNLOAD_ENV, raising=False)
    monkeypatch.setenv(LOCALHOST_ONLY_ENV, "1")
    _forbid_network(monkeypatch)

    with pytest.raises(WeightPullError, match="localhost-only"):
        refuse_weight_network_pull(CDN_URL)

    refuse_weight_network_pull("http://127.0.0.1:9/tiny.pt")


def test_cache_hit_is_not_a_download(monkeypatch, tmp_path):
    monkeypatch.setenv(NO_WEIGHT_DOWNLOAD_ENV, "1")
    monkeypatch.setenv(LOCALHOST_ONLY_ENV, "1")
    _forbid_network(monkeypatch)

    payload = b"local-checkpoint-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    url = f"https://openaipublic.azureedge.net/main/whisper/models/{digest}/tiny.pt"
    (tmp_path / "tiny.pt").write_bytes(payload)

    result = whisper._download(url, str(tmp_path), in_memory=False)
    assert result == str(tmp_path / "tiny.pt")
    assert os.environ[NO_WEIGHT_DOWNLOAD_ENV] == "1"
