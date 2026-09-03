"""Offline system-dependency smoke: ffmpeg + loopback bind. No weights."""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import pytest

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "scripts" / "system_deps_smoke.py"
BIND_PY = ROOT / "whisper" / "bind.py"
FIXTURE = ROOT / "tests" / "jfk.flac"
WILDCARD = "0.0.0.0"
BIND = "127.0.0.1"


def _mod(path):
    return runpy.run_path(str(path))


@pytest.fixture
def offline_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")

    real_urlopen = urllib.request.urlopen

    def guarded(url, *args, **kwargs):
        target = url if isinstance(url, str) else getattr(url, "full_url", str(url))
        host = urlparse(target).hostname
        if host != BIND:
            raise AssertionError(
                "offline smoke attempted non-loopback fetch: {0}".format(target)
            )
        return real_urlopen(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded)
    return tmp_path


def _no_weights(tmp_path):
    hits = []
    for pattern in ("*.pt", "*.pth", "*.safetensors", "*.bin", "*.onnx"):
        hits.extend(tmp_path.rglob(pattern))
    assert hits == [], "weight artifacts written during smoke: {0}".format(hits)


def test_bind_rejects_non_loopback():
    bind = _mod(BIND_PY)
    with pytest.raises(bind["BindError"], match="127.0.0.1"):
        bind["require_bind_127_0_0_1"](WILDCARD)
    with pytest.raises(bind["BindError"], match="127.0.0.1"):
        bind["require_bind_127_0_0_1"]("8.8.8.8")
    assert bind["require_bind_127_0_0_1"](BIND) == BIND


def test_ffmpeg_decodes_local_fixture(offline_env):
    smoke = _mod(SMOKE)
    pcm = smoke["decode_fixture"](FIXTURE)
    assert smoke["require_ffmpeg"]()
    assert len(pcm) > 16000 * 2 * 10
    assert len(pcm) < 16000 * 2 * 12
    _no_weights(offline_env)


def test_make_server_binds_loopback_only(offline_env):
    smoke = _mod(SMOKE)
    with pytest.raises(smoke["BindError"], match="127.0.0.1"):
        smoke["make_server"](WILDCARD, 0)
    server = smoke["make_server"](BIND, 0)
    try:
        host, port = server.server_address
        assert host == BIND
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        with urllib.request.urlopen(
            "http://127.0.0.1:{0}/".format(port), timeout=5
        ) as resp:
            body = json.loads(resp.read())
        assert body["ok"] is True
        assert body["bind"] == BIND
        assert body["ffmpeg"] is True
        assert body["weights"] is False
        assert body["hub"] is False
        assert body["offline"] is True
    finally:
        server.shutdown()
        server.server_close()
    _no_weights(offline_env)


def test_cli_check_is_offline_and_loopback(offline_env):
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(offline_env)
    proc = subprocess.run(
        [sys.executable, str(SMOKE), "--check", "--host", BIND],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "SMOKE OK" in proc.stdout
    assert "127.0.0.1" in proc.stdout
    _no_weights(offline_env)


def test_cli_rejects_wildcard_host(offline_env):
    proc = subprocess.run(
        [sys.executable, str(SMOKE), "--serve", "--host", WILDCARD, "--port", "0"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=10,
        env=os.environ.copy(),
    )
    assert proc.returncode == 2
    assert "127.0.0.1" in proc.stderr
    _no_weights(offline_env)


def test_start_script_is_loopback_only():
    start = ROOT / ".cursor" / "start.sh"
    text = start.read_text()
    assert "--host 127.0.0.1" in text
    assert "system_deps_smoke.py" in text
    assert WILDCARD not in text
    assert "load_model" not in text


def test_smoke_sources_never_fetch_weights_or_hub():
    smoke_src = SMOKE.read_text().lower()
    bind_src = BIND_PY.read_text().lower()
    combined = smoke_src + bind_src
    assert "load_model" not in combined
    assert "_download" not in combined
    assert "huggingface" not in combined
    assert "azureedge" not in combined
    assert "openaipublic" not in combined
    assert "field-brain" not in combined
    assert "api_key" not in combined


def test_ci_job_runs_offline_smoke():
    text = (ROOT / ".github" / "workflows" / "test.yml").read_text()
    assert "system-deps-smoke" in text
    assert "tests/test_system_deps_smoke.py" in text
    assert "HF_HUB_OFFLINE" in text
    assert "system_deps_smoke.py --check" in text
    assert (
        "test_transcribe"
        not in text.split("system-deps-smoke:")[1].split("whisper-test:")[0]
    )
