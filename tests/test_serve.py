import json
import threading
from urllib.request import urlopen

import pytest

from whisper.bind import BindError
from whisper.serve import create_server, main

ALL_INTERFACES = ".".join(("0", "0", "0", "0"))


def test_create_server_binds_127_0_0_1():
    httpd = create_server(host="127.0.0.1", port=0)
    try:
        host, port = httpd.server_address[:2]
        assert host == "127.0.0.1"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["status"] == "ok"
        assert payload["bind"] == "127.0.0.1"
        assert payload["device"] == "cpu"
        assert payload["weights"] is False
        assert payload["hub"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_create_server_refuses_all_interfaces():
    with pytest.raises(BindError, match="127.0.0.1"):
        create_server(host=ALL_INTERFACES, port=0)


def test_main_refuses_empty_host():
    assert main(["--host", "", "--port", "0"]) == 2


def test_main_refuses_all_interfaces():
    assert main(["--host", ALL_INTERFACES, "--port", "0"]) == 2
