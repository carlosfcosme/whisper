import json
import threading
import urllib.request

from whisper.bind import LOOPBACK_HOST
from whisper.serve import create_server


def test_runtime_health_binds_loopback_only():
    httpd = create_server(port=0)
    host, port = httpd.server_address[:2]
    assert host == LOOPBACK_HOST
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:{}/health".format(port), timeout=2
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["status"] == "ok"
        assert body["bind"] == "127.0.0.1"
        assert body["weights"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()
