"""Network calls that are not loopback must fail in tests."""

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

HUB_AND_WEIGHT_URLS = (
    "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin",
    "https://hf.co/openai/whisper-tiny",
    "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt",
)


@pytest.mark.parametrize("url", HUB_AND_WEIGHT_URLS)
def test_urlopen_blocks_weight_and_hub_hosts(url):
    with pytest.raises(RuntimeError, match="network"):
        urllib.request.urlopen(url, timeout=1)


def test_explicit_urlopen_monkeypatch_fails(monkeypatch):
    def _fail(url, *args, **kwargs):
        raise RuntimeError("network-call monkeypatch: weight fetch forbidden")

    monkeypatch.setattr(urllib.request, "urlopen", _fail)
    with pytest.raises(RuntimeError, match="monkeypatch"):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt",
            timeout=1,
        )


def test_urlopen_allows_loopback_health():
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = json.dumps({"bind": "127.0.0.1", "weights": False}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = httpd.server_address[:2]
        assert host == "127.0.0.1"
        with urllib.request.urlopen("http://127.0.0.1:%s/" % port, timeout=2) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["bind"] == "127.0.0.1"
        assert body["weights"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()
