import json
import threading
from urllib.request import urlopen

import pytest

from whisper.runtime import BindError
from whisper.serve import create_server, main


def test_create_server_binds_127_0_0_1():
    httpd = create_server("127.0.0.1", 0)
    try:
        host, port = httpd.server_address[:2]
        assert host == "127.0.0.1"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        with urlopen("http://127.0.0.1:{}/health".format(port), timeout=2) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["status"] == "ok"
        assert body["bind"] == "127.0.0.1"
        assert body["weights"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.parametrize("host", ["0.0.0.0", "", "::", "8.8.8.8"])
def test_create_server_refuses_non_localhost(host):
    with pytest.raises(BindError, match="127.0.0.1"):
        create_server(host, 0)


def test_cli_refuses_wildcard_host():
    assert main(["--host", "0.0.0.0", "--port", "0"]) == 2
