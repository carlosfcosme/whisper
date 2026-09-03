import json
import threading
import urllib.request

import pytest

from whisper.bind import BIND_HOST, BindError, require_bind_127_0_0_1
from whisper.serve import make_server

pytestmark = pytest.mark.localhost_only

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
        "8.8.8.8",
        "",
        None,
    ],
)
def test_require_bind_rejects_non_127(host):
    with pytest.raises(BindError, match="127.0.0.1"):
        require_bind_127_0_0_1(host)


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


def test_serve_cli_rejects_wildcard():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "whisper", "serve", "--host", WILDCARD, "--port", "0"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "127.0.0.1" in (result.stderr + result.stdout)


def test_serve_module_has_no_weight_or_hub_path():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "whisper" / "serve.py").read_text()
    assert "load_model" not in source
    assert "_download" not in source
    assert "huggingface" not in source
    assert "azureedge" not in source
    assert WILDCARD not in source
