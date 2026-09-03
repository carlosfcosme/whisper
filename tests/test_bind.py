import json
import threading
import urllib.request
from pathlib import Path

import pytest

from whisper.bind import BIND_HOST, BindError, require_bind_127_0_0_1
from whisper.serve import make_server

REPO_ROOT = Path(__file__).resolve().parents[1]
WILDCARD = "0.0.0.0"


def _start_scripts():
    paths = []
    paths.extend(sorted(REPO_ROOT.glob(".cursor/*.sh")))
    paths.extend(sorted(REPO_ROOT.glob("start*.sh")))
    paths.extend(sorted(REPO_ROOT.glob("serve*.sh")))
    env = REPO_ROOT / ".cursor" / "environment.json"
    if env.exists():
        paths.append(env)
    return paths


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
    assert start.is_file()
    text = start.read_text()
    assert "--host 127.0.0.1" in text
    assert "whisper.serve" in text
    assert WILDCARD not in text


def test_start_scripts_forbid_wildcard_bind():
    hits = []
    for path in _start_scripts():
        if WILDCARD in path.read_text():
            hits.append(str(path.relative_to(REPO_ROOT)))
    assert hits == [], f"{WILDCARD} must not appear in start scripts: {hits}"


def test_serve_module_has_no_weight_or_hub_path():
    source = (REPO_ROOT / "whisper" / "serve.py").read_text()
    assert "load_model" not in source
    assert "_download" not in source
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
        assert body["device"] == "cpu"
        assert body["weights"] is False
        assert body["hub"] is False
    finally:
        server.shutdown()
        server.server_close()
