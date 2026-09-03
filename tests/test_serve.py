import json
import threading
from urllib.request import urlopen

import pytest

from whisper.bind import BindError
from whisper.serve import create_server, main


def test_create_server_rejects_all_interfaces():
    with pytest.raises(BindError):
        create_server("0.0.0.0", 0)


def test_create_server_binds_loopback_and_health():
    server = create_server("127.0.0.1", 0)
    host, port = server.server_address[:2]
    assert host == "127.0.0.1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["status"] == "ok"
        assert payload["host"] == "127.0.0.1"
        assert payload["weights"] is False
        assert payload["hub"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_cli_rejects_0_0_0_0():
    assert main(("--host", "0.0.0.0", "--port", "0")) == 2
