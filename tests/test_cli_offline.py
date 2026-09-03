"""CLI offline tests: fail on download or non-loopback bind.

Uses only local fixtures (``jfk.flac`` and a runtime toy checkpoint).
Never fetches Hub/Azure weights.
"""

import importlib.util
import json
import os
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

import pytest

from whisper.localhost import ALL_INTERFACES, LOOPBACK_BIND, BindError
from whisper.serve import create_server
from whisper.serve import main as serve_main
from whisper.transcribe import cli

_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "local_fixtures", Path(__file__).resolve().parent / "local_fixtures.py"
)
_FIXTURES = importlib.util.module_from_spec(_FIXTURE_SPEC)
assert _FIXTURE_SPEC.loader is not None
_FIXTURE_SPEC.loader.exec_module(_FIXTURES)
JFK_FLAC = _FIXTURES.JFK_FLAC
toy_checkpoint = _FIXTURES.toy_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cli_env(tmp_path):
    """Isolate caches only. Keep HOME so user-site torch stays importable."""
    xdg = tmp_path / "xdg"
    hf = tmp_path / "hf"
    xdg.mkdir(exist_ok=True)
    hf.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(xdg)
    env["HF_HOME"] = str(hf)
    env["HF_HUB_CACHE"] = str(hf / "hub")
    env["WHISPER_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env.pop("WHISPER_ALLOW_DOWNLOADS", None)
    return env, xdg


def _run_cli(args, tmp_path):
    env, xdg = _cli_env(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "whisper", *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, xdg


def _weight_files(root: Path):
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pt"]


def test_local_audio_fixture_exists():
    assert JFK_FLAC.is_file()
    assert JFK_FLAC.suffix == ".flac"


def test_cli_help_uses_no_weights_and_no_bind(tmp_path):
    result, xdg = _run_cli(["--help"], tmp_path)
    assert result.returncode == 0
    assert "--allow_downloads" in result.stdout
    assert _weight_files(xdg) == []


def test_cli_named_model_fails_without_download(tmp_path):
    """Named checkpoints must not be fetched. Local fixtures only."""
    out = tmp_path / "out"
    result, xdg = _run_cli(
        [
            str(JFK_FLAC),
            "--model",
            "tiny",
            "--device",
            "cpu",
            "--output_dir",
            str(out),
            "--output_format",
            "txt",
        ],
        tmp_path,
    )
    assert result.returncode != 0
    text = result.stdout + result.stderr
    assert "offline" in text.lower() or "refusing" in text.lower()
    assert _weight_files(xdg) == []
    assert _weight_files(out) == []


def test_cli_local_fixture_does_not_download(tmp_path, monkeypatch):
    ckpt = toy_checkpoint(tmp_path / "fixtures" / "toy.pt")
    out = tmp_path / "out"
    attempted = []

    def _boom(url, *args, **kwargs):
        attempted.append(url)
        raise AssertionError("CLI must not download: {}".format(url))

    monkeypatch.setattr(urllib.request, "urlopen", _boom)

    def _fake_transcribe(model, audio_path, **kwargs):
        assert Path(audio_path) == JFK_FLAC
        return {"text": "local fixture", "segments": [], "language": "en"}

    # whisper.transcribe is the function re-exported from the package.
    monkeypatch.setattr(
        sys.modules["whisper.transcribe"], "transcribe", _fake_transcribe
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "whisper",
            str(JFK_FLAC),
            "--model",
            str(ckpt),
            "--device",
            "cpu",
            "--output_dir",
            str(out),
            "--output_format",
            "txt",
        ],
    )
    cli()
    assert attempted == []
    assert (out / "jfk.txt").is_file()
    assert (
        _weight_files(tmp_path / "xdg") == [] if (tmp_path / "xdg").exists() else True
    )


def test_cli_serve_rejects_all_interfaces(tmp_path):
    result, xdg = _run_cli(["serve", "--host", ALL_INTERFACES, "--port", "0"], tmp_path)
    assert result.returncode == 2
    assert LOOPBACK_BIND in result.stderr
    assert ALL_INTERFACES == "0.0.0.0"
    assert _weight_files(xdg) == []


def test_cli_serve_rejects_lan_and_wildcard(tmp_path):
    for host in ("*", "::", "192.168.1.10", "example.com"):
        result, _xdg = _run_cli(["serve", "--host", host, "--port", "0"], tmp_path)
        assert result.returncode == 2, host
        assert LOOPBACK_BIND in result.stderr


def test_cli_serve_binds_loopback_only(tmp_path):
    httpd = create_server(host=LOOPBACK_BIND, port=0)
    try:
        bound_host, bound_port = httpd.server_address[:2]
        assert bound_host == LOOPBACK_BIND
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        with urllib.request.urlopen(
            "http://127.0.0.1:{}/health".format(bound_port)
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["status"] == "ok"
        assert payload["bind"] == LOOPBACK_BIND
        assert payload["weights"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_create_server_refuses_non_loopback():
    with pytest.raises(BindError):
        create_server(host=ALL_INTERFACES, port=0)


def test_serve_main_rejects_zero_addr():
    assert serve_main(["--host", ALL_INTERFACES, "--port", "0"]) == 2


def test_start_script_is_loopback_only():
    start = REPO_ROOT / ".cursor" / "start.sh"
    assert start.is_file()
    text = start.read_text(encoding="utf-8")
    assert "--host 127.0.0.1" in text
    assert ALL_INTERFACES not in text
