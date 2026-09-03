import json
import threading
import urllib.request

import pytest

from whisper.runtime import BindError
from whisper.serve import make_server


def test_health_binds_127_0_0_1():
    httpd = make_server(port=0)
    host, port = httpd.server_address[:2]
    assert host == "127.0.0.1"
    assert port > 0
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{port}/health"
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["ok"] is True
        assert payload["bind"] == "127.0.0.1"
        assert payload["device"] == "cpu"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_make_server_refuses_wildcard():
    with pytest.raises(BindError, match="0.0.0.0"):
        make_server(host="0.0.0.0", port=0)
