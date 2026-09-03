"""Live whisper.serve on 127.0.0.1. No weight download."""

import json
import threading
import urllib.request

import pytest

from whisper.bind import BindError
from whisper.serve import create_server, main


def test_create_server_refuses_empty_and_all_interfaces():
    with pytest.raises(BindError, match="required"):
        create_server(host="")
    with pytest.raises(BindError):
        create_server(host=".".join(("0", "0", "0", "0")))


def test_health_endpoint_on_127_0_0_1():
    httpd = create_server(host="127.0.0.1", port=0)
    host, port = httpd.server_address[:2]
    assert host == "127.0.0.1"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["status"] == "ok"
        assert payload["bind"] == "127.0.0.1"
        assert payload["weights"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_main_refuses_empty_host():
    assert main(["--host", "", "--port", "0"]) == 2
