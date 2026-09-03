import json
import subprocess
import sys
import threading
import urllib.request

from whisper.serve import create_server


def test_health_binds_127_0_0_1():
    httpd = create_server(host="127.0.0.1", port=0)
    host, port = httpd.server_address[:2]
    assert host == "127.0.0.1"
    assert port > 0
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["status"] == "ok"
        assert payload["bind"] == "127.0.0.1"
        assert payload["device"] == "cpu"
        assert payload["weights"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_module_serve_refuses_wildcard():
    result = subprocess.run(
        [sys.executable, "-m", "whisper.serve", "--host", "0.0.0.0", "--port", "0"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "0.0.0.0" in result.stderr
