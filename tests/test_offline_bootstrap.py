import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

import whisper
from whisper.bind import require_loopback_host
from whisper.bootstrap import offline_bootstrap
from whisper.serve import make_server

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_ignored_artifacts", ROOT / "scripts" / "check_ignored_artifacts.py"
)
check_ignored = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_ignored)


def test_offline_bootstrap_does_not_fetch_or_store(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    info = offline_bootstrap()
    assert info["device"] == "cpu"
    assert info["bind"] == "127.0.0.1"
    assert info["offline"] is True
    assert info["no_store"] is True
    assert info["weights"] is False
    assert info["fetched"] is False
    with pytest.raises(RuntimeError, match="offline|no-store"):
        whisper.load_model("tiny")
    assert list(tmp_path.rglob("*.pt")) == []
    assert list(tmp_path.rglob("*.safetensors")) == []


def test_offline_bootstrap_requires_loopback_bind():
    with pytest.raises(ValueError, match="127.0.0.1"):
        require_loopback_host("0.0.0.0")
    info = offline_bootstrap()
    httpd = make_server(info["bind"], 0)
    try:
        host, _port = httpd.server_address[:2]
        sockname = httpd.socket.getsockname()[0]
        assert host == "127.0.0.1"
        assert sockname == "127.0.0.1"
        assert host != "0.0.0.0"
        assert sockname != "0.0.0.0"
    finally:
        httpd.server_close()


def test_bootstrap_cli_json_no_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-m", "whisper.bootstrap"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["bind"] == "127.0.0.1"
    assert payload["fetched"] is False
    assert list(tmp_path.rglob("*")) == []


def test_ignored_artifacts_are_not_tracked():
    assert check_ignored.missing_ignore_rules() == []
    assert check_ignored.tracked_artifacts() == []
    assert check_ignored.plant_and_verify() == []
    assert check_ignored.main() == 0
