"""CI/offline must not WAN-pull weights; services bind loopback only."""

import hashlib
import socket
import urllib.request
from pathlib import Path

import pytest

import whisper

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHT_URL = (
    "https://openaipublic.azureedge.net/main/whisper/models/"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/tiny.pt"
)


def test_offline_download_refuses_wan_fetch(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="WHISPER_OFFLINE"):
        whisper._download(WEIGHT_URL, str(tmp_path), in_memory=False)
    assert list(tmp_path.glob("*.pt")) == []


def test_offline_download_uses_cached_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    payload = b"cached-checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    url = "https://example.invalid/whisper/models/%s/tiny.pt" % digest
    target = tmp_path / "tiny.pt"
    target.write_bytes(payload)
    assert whisper._download(url, str(tmp_path), in_memory=False) == str(target)


def test_load_model_offline_does_not_fetch(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="WHISPER_OFFLINE"):
        whisper.load_model("tiny", download_root=str(tmp_path))
    assert list(tmp_path.rglob("*.pt")) == []


def test_urlopen_hook_blocks_weight_urls():
    with pytest.raises(RuntimeError, match="WAN model weight download"):
        urllib.request.urlopen(WEIGHT_URL)


def test_package_source_does_not_bind_all_interfaces():
    offenders = []
    for path in (REPO_ROOT / "whisper").rglob("*.py"):
        text = path.read_text()
        if "0.0.0.0" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], "application code must not bind 0.0.0.0: %s" % offenders


def test_bind_all_interfaces_is_refused():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="127.0.0.1"):
            sock.bind(("0.0.0.0", 0))
    finally:
        sock.close()


def test_bind_loopback_is_allowed():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()
        assert host == "127.0.0.1"
        assert port > 0
    finally:
        sock.close()


def test_environment_publishes_no_ports_or_keys():
    env = (REPO_ROOT / ".cursor" / "environment.json").read_text()
    install = (REPO_ROOT / ".cursor" / "install.sh").read_text()
    assert '"ports"' not in env
    assert "0.0.0.0" not in env
    assert "load_model" not in install
    assert "_download" not in install
    assert "azureedge" not in install
    assert "API_KEY" not in install
    assert "SECRET" not in install
    assert "API_KEY" not in env


def test_no_secret_or_weight_files_tracked():
    import subprocess

    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", ".env", ".env.*", "*.pt", "*.pth"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert listed.returncode == 0, listed.stderr
    tracked = [path for path in listed.stdout.split("\0") if path]
    assert tracked == [], "secrets/weights must stay untracked: %s" % tracked
