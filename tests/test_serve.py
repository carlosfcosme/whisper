import json
import threading
import urllib.request
from pathlib import Path

import pytest

from whisper.serve import (
    ALL_INTERFACES,
    LOOPBACK_BIND,
    BindError,
    create_server,
    is_loopback_host,
    listen_url,
    main,
    require_loopback_bind,
    serve,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _discover_start_scripts(root: Path):
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


def _start_script_all_interface_hits(paths):
    hits = []
    for path in paths:
        if ALL_INTERFACES in path.read_text(encoding="utf-8"):
            hits.append(str(path))
    return hits


def test_require_loopback_bind_defaults_to_127():
    assert LOOPBACK_BIND == "127.0.0.1"
    assert require_loopback_bind() == "127.0.0.1"
    assert require_loopback_bind("localhost") == "127.0.0.1"
    assert require_loopback_bind("127.0.0.1") == "127.0.0.1"
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")


@pytest.mark.parametrize(
    "host", [ALL_INTERFACES, "::", "*", "192.168.1.1", "example.com", "10.0.0.1"]
)
def test_require_loopback_bind_rejects_non_localhost(host):
    with pytest.raises(BindError, match="127.0.0.1"):
        require_loopback_bind(host)
    assert not is_loopback_host(host)


def test_serve_listens_on_127_0_0_1():
    httpd = serve(port=0)
    try:
        host, port = httpd.server_address[:2]
        assert host == "127.0.0.1"
        assert is_loopback_host(host)
        assert port > 0

        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        with urllib.request.urlopen(
            "http://127.0.0.1:{}/health".format(port), timeout=2
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["status"] == "ok"
        assert body["bind"] == "127.0.0.1"
        assert body["device"] == "cpu"
        assert body["hub"] is False
        assert body["weights"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_create_server_rejects_all_interfaces():
    with pytest.raises(BindError, match="127.0.0.1"):
        create_server(host=ALL_INTERFACES, port=0)


def test_serve_rejects_wildcard_host():
    with pytest.raises(BindError, match="127.0.0.1"):
        serve(host=ALL_INTERFACES, port=0)


def test_cli_refuses_all_interfaces(capsys):
    code = main(["--host", ALL_INTERFACES, "--port", "0"])
    assert code == 2
    err = capsys.readouterr().err
    assert "127.0.0.1" in err


def test_listen_url_brackets_ipv6():
    assert listen_url("::1", 8765) == "http://[::1]:8765"
    assert listen_url("[::1]", 80) == "http://[::1]:80"
    assert listen_url(LOOPBACK_BIND, 8765) == "http://127.0.0.1:8765"


def test_start_script_exists_and_uses_loopback():
    start = REPO_ROOT / ".cursor" / "start.sh"
    assert start.is_file()
    text = start.read_text(encoding="utf-8")
    assert "127.0.0.1" in text
    assert "whisper.serve" in text
    assert ALL_INTERFACES not in text


def test_repo_start_scripts_do_not_bind_all_interfaces():
    scripts = _discover_start_scripts(REPO_ROOT)
    assert any(path.name == "start.sh" for path in scripts)
    hits = _start_script_all_interface_hits(scripts)
    assert hits == [], "{} is not allowed in start scripts: {}".format(
        ALL_INTERFACES, hits
    )
