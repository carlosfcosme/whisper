import json
import threading
import urllib.request

import pytest

from whisper.bind import LOOPBACK_HOST, BindError
from whisper.serve import create_server, main

ALL_INTERFACES = ".".join(("0", "0", "0", "0"))


def test_create_server_binds_loopback():
    httpd = create_server(LOOPBACK_HOST, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = httpd.server_address[:2]
        assert host == LOOPBACK_HOST
        url = "http://{}:{}/health".format(host, port)
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["status"] == "ok"
        assert payload["bind"] == LOOPBACK_HOST
        assert payload["weights"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_create_server_refuses_all_interfaces():
    with pytest.raises(BindError):
        create_server(ALL_INTERFACES, port=0)


def test_main_refuses_empty_host():
    assert main(["--host", "", "--port", "0"]) == 2


def test_serve_source_has_no_load_model():
    import ast
    import inspect

    import whisper.serve as serve

    tree = ast.parse(inspect.getsource(serve))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "load_model" not in names
    assert "huggingface" not in inspect.getsource(serve).lower()
