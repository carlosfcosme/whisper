"""Focused offline pytest: fetches fail, 127.0.0.1 bind, local fixtures.

No unshare. No Hub. No weight download. Bind-and-weights CI can run this
without torch (model-fetch cases importorskip).
"""

from __future__ import annotations

import importlib.util
import json
import socket
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

pytestmark = pytest.mark.offline

REPO = Path(__file__).resolve().parents[1]
ALL_INTERFACES = ".".join(("0", "0", "0", "0"))
HUB_URL = "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"
CDN_URL = (
    "https://openaipublic.azureedge.net/main/whisper/models/"
    "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _OkHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"status":"ok","bind":"127.0.0.1"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


def test_network_monkeypatch_fails_except_loopback(isolated_cache, loopback_bind):
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen(HUB_URL)
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen(CDN_URL)
    with pytest.raises(RuntimeError, match="forbidden"):
        socket.create_connection(("huggingface.co", 443), timeout=1)
    assert loopback_bind == "127.0.0.1"
    assert list(isolated_cache.iterdir()) == []


def test_model_fetch_monkeypatch_fails(fail_model_fetch, isolated_cache):
    whisper = pytest.importorskip("whisper")
    with pytest.raises(RuntimeError, match="forbidden"):
        whisper._download(CDN_URL, str(isolated_cache), in_memory=False)
    with pytest.raises(RuntimeError, match="forbidden"):
        whisper.load_model("tiny", download_root=str(isolated_cache))
    leftover = [
        path
        for path in isolated_cache.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".bin"}
    ]
    assert leftover == []


def test_offline_policy_refuses_fetch_without_torch(isolated_cache):
    offline = _load("whisper_offline_focused", REPO / "whisper" / "offline.py")
    with pytest.raises(offline.WeightDownloadError, match="Hub"):
        offline.refuse_weight_network_pull(HUB_URL)
    with pytest.raises(offline.WeightDownloadError, match="disabled"):
        offline.refuse_weight_network_pull(CDN_URL)
    assert list(isolated_cache.iterdir()) == []


def test_bind_requires_loopback_fixture(loopback_bind):
    bind = _load("whisper_bind_focused", REPO / "whisper" / "bind.py")
    assert bind.require_bind_127_0_0_1(None) == loopback_bind
    assert bind.require_bind_127_0_0_1("localhost") == loopback_bind
    assert bind.require_bind_127_0_0_1(loopback_bind) == loopback_bind
    with pytest.raises(bind.BindError, match="required"):
        bind.require_bind_127_0_0_1("")
    with pytest.raises(bind.BindError):
        bind.require_bind_127_0_0_1(ALL_INTERFACES)


def test_loopback_http_uses_local_bind(loopback_bind):
    bind = _load("whisper_bind_focused_httpd", REPO / "whisper" / "bind.py")
    httpd = bind.create_loopback_httpd(_OkHandler, host=loopback_bind, port=0)
    host, port = httpd.server_address[:2]
    assert host == loopback_bind
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen("http://%s:%s/" % (loopback_bind, port)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["status"] == "ok"
        assert payload["bind"] == loopback_bind
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_start_sh_hardcodes_loopback_fixture(loopback_bind):
    text = (REPO / ".cursor" / "start.sh").read_text(encoding="utf-8")
    assert "--host %s" % loopback_bind in text
    assert ALL_INTERFACES not in text


def test_local_audio_fixture_is_on_disk(local_audio, isolated_cache):
    assert local_audio.is_file()
    assert local_audio.suffix == ".flac"
    assert local_audio.stat().st_size > 0
    assert list(isolated_cache.iterdir()) == []
