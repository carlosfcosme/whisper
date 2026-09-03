"""Localhost bind policy. Does not download model weights or use secrets."""

import json
import threading
import urllib.request
from pathlib import Path
from typing import Iterable, List

import pytest

from whisper.localhost import (
    ALL_INTERFACES,
    LOOPBACK_BIND,
    BindError,
    is_loopback_host,
    require_loopback_bind,
)
from whisper.serve import create_server, main

REPO_ROOT = Path(__file__).resolve().parents[1]


def discover_start_scripts(root: Path) -> List[Path]:
    found = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        name = path.name
        if name == "start.sh" or (name.startswith("start-") and name.endswith(".sh")):
            found.append(path)
    env = root / ".cursor" / "environment.json"
    if env.is_file():
        found.append(env)
    return sorted(set(found))


def start_script_all_interface_hits(paths: Iterable[Path]) -> List[str]:
    hits = []
    for path in paths:
        if ALL_INTERFACES in path.read_text(encoding="utf-8"):
            hits.append(str(path))
    return hits


def assert_start_scripts_localhost_only(paths: Iterable[Path]) -> None:
    hits = start_script_all_interface_hits(paths)
    assert hits == [], f"{ALL_INTERFACES} is not allowed in start scripts: {hits}"


@pytest.mark.parametrize("host", [LOOPBACK_BIND, "localhost", "LOCALHOST", "::1"])
def test_require_loopback_bind_allows_loopback(host):
    bound = require_loopback_bind(host)
    assert is_loopback_host(bound)
    if host.lower() == "localhost" or host == LOOPBACK_BIND:
        assert bound == LOOPBACK_BIND


@pytest.mark.parametrize(
    "host",
    [ALL_INTERFACES, "::", "*", "", "192.168.1.10", "example.com", "10.0.0.1"],
)
def test_require_loopback_bind_refuses_non_loopback(host):
    with pytest.raises(BindError):
        require_loopback_bind(host)


def test_require_loopback_bind_rejects_all_interfaces():
    """Headline contract: 0.0.0.0 is never a valid bind host."""
    with pytest.raises(BindError, match="127.0.0.1"):
        require_loopback_bind(ALL_INTERFACES)
    assert ALL_INTERFACES == "0.0.0.0"
    assert not is_loopback_host(ALL_INTERFACES)


def test_create_server_refuses_all_interfaces():
    with pytest.raises(BindError):
        create_server(host=ALL_INTERFACES, port=0)


def test_create_server_binds_loopback_only():
    httpd = create_server(host=LOOPBACK_BIND, port=0)
    try:
        host, port = httpd.server_address[:2]
        assert host == LOOPBACK_BIND
        assert is_loopback_host(host)
        assert port > 0
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=2
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["status"] == "ok"
        assert body["bind"] == LOOPBACK_BIND
        assert body["weights"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_cli_refuses_all_interfaces(capsys):
    code = main(["--host", ALL_INTERFACES, "--port", "0"])
    assert code == 2
    err = capsys.readouterr().err
    assert "127.0.0.1" in err
    assert ALL_INTERFACES in err or "all-interfaces" in err


def test_start_script_exists_and_uses_loopback():
    start = REPO_ROOT / ".cursor" / "start.sh"
    assert start.is_file()
    text = start.read_text(encoding="utf-8")
    assert "127.0.0.1" in text
    assert "whisper.serve" in text
    assert ALL_INTERFACES not in text


def test_repo_start_scripts_do_not_bind_all_interfaces():
    scripts = discover_start_scripts(REPO_ROOT)
    assert any(p.name == "start.sh" for p in scripts)
    assert_start_scripts_localhost_only(scripts)


def test_scan_fails_when_all_interfaces_in_start_script(tmp_path):
    script = tmp_path / "start.sh"
    script.write_text(f"python3 -m http.server --bind {ALL_INTERFACES}\n")
    with pytest.raises(AssertionError, match="not allowed in start scripts"):
        assert_start_scripts_localhost_only([script])
