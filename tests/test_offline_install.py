"""Offline install: block network/model downloads, loopback bind, ignored weights."""

import subprocess
import sys
import threading
from pathlib import Path

import pytest

import whisper
from whisper.defaults import (
    DEFAULT_HOST,
    WEIGHT_SUFFIXES,
    downloads_blocked,
    require_loopback_host,
)
from whisper.local_server import create_server

ROOT = Path(__file__).resolve().parents[1]
IGNORE_SCRIPT = ROOT / ".github" / "scripts" / "validate_ignored_weights.sh"


@pytest.fixture(autouse=True)
def block_network_and_model_downloads(monkeypatch, tmp_path):
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    def blocked(url, *args, **kwargs):
        raise RuntimeError(f"network/model download blocked: {url}")

    monkeypatch.setattr("urllib.request.urlopen", blocked)


def test_downloads_blocked_env():
    assert downloads_blocked() is True


def test_offline_pip_install_has_no_weights(tmp_path):
    dest = tmp_path / "site"
    dest.mkdir()
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--no-build-isolation",
            "--target",
            str(dest),
            str(ROOT),
        ]
    )
    installed = []
    for suffix in WEIGHT_SUFFIXES:
        installed.extend(dest.rglob(f"*{suffix}"))
    assert installed == []
    assert (dest / "whisper" / "defaults.py").is_file()


def test_model_download_is_blocked(tmp_path):
    cache = tmp_path / "xdg" / "whisper"
    azure = (
        "https://openaipublic.azureedge.net/main/whisper/models/"
        "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef/tiny.pt"
    )
    with pytest.raises(RuntimeError, match="offline: model download blocked"):
        whisper._download(azure, str(cache), False)
    assert list(tmp_path.rglob("*.pt")) == []

    with pytest.raises(ValueError, match="Hugging Face Hub"):
        whisper._download(
            "https://huggingface.co/openai/whisper-tiny", str(cache), False
        )
    with pytest.raises(FileNotFoundError, match="local checkpoint"):
        whisper.resolve_local_checkpoint("tiny")
    with pytest.raises(ValueError, match="Hugging Face Hub"):
        whisper.load_model(
            "https://huggingface.co/openai/whisper-tiny", local_only=True
        )


def test_offline_server_binds_loopback_only():
    require_loopback_host("127.0.0.1")
    with pytest.raises(ValueError, match="127.0.0.1"):
        require_loopback_host("0.0.0.0")
    with pytest.raises(ValueError, match="127.0.0.1"):
        create_server("0.0.0.0", 0)

    server = create_server(DEFAULT_HOST, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, _port = server.socket.getsockname()[:2]
        assert host == "127.0.0.1"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_ignored_weights_are_not_tracked():
    subprocess.check_call(["bash", str(IGNORE_SCRIPT)], cwd=str(ROOT))
    dummy = ROOT / ".offline-ignored-weight.pt"
    assert not dummy.exists()
    tracked = subprocess.check_output(
        ["git", "ls-files", "--", "*.pt", "*.pth", "*.ckpt", "*.safetensors"],
        cwd=str(ROOT),
        text=True,
    ).strip()
    assert tracked == ""


def test_ci_runs_offline_install_job():
    yml = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    assert "offline-install" in yml
    assert "WHISPER_OFFLINE" in yml
    assert "validate_ignored_weights.sh" in yml
    assert "--no-index" in yml
    assert "--no-deps" in yml
    assert "--no-build-isolation" in yml
