import json
import threading
import urllib.request
from pathlib import Path

import pytest

from whisper.bind import BIND_HOST, BindError, require_bind_127_0_0_1
from whisper.serve import make_server

pytestmark = pytest.mark.localhost_only

REPO_ROOT = Path(__file__).resolve().parents[1]
WILDCARD = "0.0.0.0"


def test_require_bind_accepts_loopback():
    assert require_bind_127_0_0_1(BIND_HOST) == BIND_HOST
    assert require_bind_127_0_0_1(" 127.0.0.1 ") == BIND_HOST


@pytest.mark.parametrize(
    "host",
    [
        WILDCARD,
        "::",
        "::1",
        "localhost",
        "127.0.0.2",
        "10.0.0.1",
        "192.168.1.1",
        "8.8.8.8",
        "openaipublic.azureedge.net",
        "huggingface.co",
        "",
        None,
    ],
)
def test_require_bind_rejects_non_127(host):
    with pytest.raises(BindError, match="127.0.0.1"):
        require_bind_127_0_0_1(host)


def test_start_script_binds_loopback():
    start = REPO_ROOT / ".cursor" / "start.sh"
    assert start.is_file(), "missing .cursor/start.sh serve/bind path"
    text = start.read_text()
    assert "--host 127.0.0.1" in text
    assert "whisper.serve" in text
    assert WILDCARD not in text


def test_environment_start_is_localhost():
    env = json.loads((REPO_ROOT / ".cursor" / "environment.json").read_text())
    assert env.get("start") == "bash .cursor/start.sh"
    assert WILDCARD not in json.dumps(env)


def test_serve_module_has_no_weight_or_hub_path():
    source = (REPO_ROOT / "whisper" / "serve.py").read_text()
    assert "load_model" not in source
    assert "_download" not in source
    assert "openaipublic" not in source
    assert "huggingface" not in source
    assert WILDCARD not in source


def test_make_server_rejects_wildcard_before_bind():
    with pytest.raises(BindError, match="127.0.0.1"):
        make_server(WILDCARD, 0)


def test_make_server_listens_on_loopback_only():
    server = make_server(BIND_HOST, 0)
    try:
        host, port = server.server_address
        assert host == BIND_HOST
        assert port > 0
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as response:
            body = json.loads(response.read())
        assert body["ok"] is True
        assert body["bind"] == BIND_HOST
        assert body["weights"] is False
        assert body["hub"] is False
    finally:
        server.shutdown()
        server.server_close()
