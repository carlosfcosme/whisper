import json
import threading
import urllib.request
from pathlib import Path

import pytest

from whisper.bind import BindError
from whisper.serve import create_server, main

ALL_INTERFACES = ".".join(("0",) * 4)
LOOPBACK = "127.0.0.1"


def test_create_server_refuses_all_interfaces():
    with pytest.raises(BindError):
        create_server(host=ALL_INTERFACES, port=0)


def test_create_server_binds_loopback_only():
    httpd = create_server(host=LOOPBACK, port=0)
    try:
        host, port = httpd.server_address[:2]
        assert host == LOOPBACK
        assert port > 0
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        url = "http://127.0.0.1:%s/health" % port
        with urllib.request.urlopen(url, timeout=2) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["status"] == "ok"
        assert body["bind"] == LOOPBACK
        assert body["device"] == "cpu"
        assert body["weights"] is False
        assert body["hub"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_cli_refuses_all_interfaces(capsys):
    code = main(["--host", ALL_INTERFACES, "--port", "0"])
    assert code == 2
    err = capsys.readouterr().err
    assert "127.0.0.1" in err


def test_start_script_binds_loopback_only():
    start = Path(__file__).resolve().parents[1] / ".cursor" / "start.sh"
    text = start.read_text(encoding="utf-8")
    assert "127.0.0.1" in text
    assert "whisper.serve" in text
    assert ALL_INTERFACES not in text
