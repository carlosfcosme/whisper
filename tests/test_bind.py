"""Tests that fail if bind is not 127.0.0.1."""

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from whisper.bind import BIND_HOST, BindError, require_bind_127_0_0_1
from whisper.serve import make_server

REPO_ROOT = Path(__file__).resolve().parents[1]
WILDCARD = "0.0.0.0"


def start_script_paths():
    paths = []
    paths.extend(sorted(REPO_ROOT.glob(".cursor/*.sh")))
    paths.extend(sorted(REPO_ROOT.glob("start*.sh")))
    paths.extend(sorted(REPO_ROOT.glob("serve*.sh")))
    paths.extend(sorted(REPO_ROOT.glob("docker-compose*.yml")))
    paths.extend(sorted(REPO_ROOT.glob("**/spark*.yml")))
    env = REPO_ROOT / ".cursor" / "environment.json"
    if env.exists():
        paths.append(env)
    return [p for p in paths if p.is_file()]


def forbidden_bind_hosts_in(text):
    hits = []
    for token in (WILDCARD, "--live true", "--live=true"):
        if token in text:
            hits.append(token)
    return hits


def test_require_bind_accepts_loopback_only():
    assert require_bind_127_0_0_1(BIND_HOST) == BIND_HOST
    assert require_bind_127_0_0_1(" 127.0.0.1 ") == BIND_HOST


@pytest.mark.parametrize(
    "host",
    [
        WILDCARD,
        "::",
        "::1",
        "localhost",
        "",
        None,
        "10.0.0.1",
        "192.168.1.1",
        "8.8.8.8",
        "huggingface.co",
    ],
)
def test_require_bind_fails_if_not_127(host):
    with pytest.raises(BindError, match="127.0.0.1"):
        require_bind_127_0_0_1(host)


def test_make_server_fails_on_wildcard_before_bind():
    with pytest.raises(BindError, match="127.0.0.1"):
        make_server(WILDCARD, 0)


def test_make_server_listens_on_127_only():
    server = make_server(BIND_HOST, 0)
    try:
        host, port = server.server_address
        assert host == BIND_HOST
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as response:
            body = json.loads(response.read())
        assert body["bind"] == BIND_HOST
        assert body["device"] == "cpu"
        assert body["weights"] is False
        assert body["hub"] is False
    finally:
        server.shutdown()
        server.server_close()


def test_start_scripts_fail_if_not_loopback():
    start = REPO_ROOT / ".cursor" / "start.sh"
    assert start.is_file()
    text = start.read_text()
    assert "--host 127.0.0.1" in text
    hits = []
    for path in start_script_paths():
        bad = forbidden_bind_hosts_in(path.read_text())
        if bad:
            hits.append((str(path.relative_to(REPO_ROOT)), bad))
    assert hits == [], f"non-loopback bind in start scripts: {hits}"


def test_scanner_fails_when_wildcard_is_injected(tmp_path):
    planted = tmp_path / "start.sh"
    planted.write_text("python3 -m whisper.serve --host 0.0.0.0 --port 80\n")
    assert forbidden_bind_hosts_in(planted.read_text()) == [WILDCARD]


def test_serve_has_no_weight_or_hub_or_wildcard():
    source = (REPO_ROOT / "whisper" / "serve.py").read_text()
    assert "load_model" not in source
    assert "_download" not in source
    assert "huggingface" not in source
    assert WILDCARD not in source
    assert "--live" not in source
