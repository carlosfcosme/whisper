import json
import threading
import urllib.request
from pathlib import Path
from typing import Iterable, List

import pytest

from whisper.serve import (
    BindError,
    create_server,
    is_loopback_host,
    main,
    normalize_bind_host,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_INTERFACES = "0.0.0.0"


def discover_start_scripts(root: Path) -> List[Path]:
    found = []
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        name = path.name
        if name == "start.sh" or (name.startswith("start-") and name.endswith(".sh")):
            found.append(path)
    env = root / ".cursor" / "environment.json"
    if env.is_file():
        found.append(env)
    return sorted(set(found))


def assert_start_scripts_localhost_only(paths: Iterable[Path]) -> None:
    hits = [str(p) for p in paths if ALL_INTERFACES in p.read_text(encoding="utf-8")]
    assert hits == [], f"{ALL_INTERFACES} is not allowed in start scripts: {hits}"


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "LOCALHOST", "::1"])
def test_normalize_bind_host_allows_loopback(host):
    assert is_loopback_host(normalize_bind_host(host))


@pytest.mark.parametrize(
    "host",
    [ALL_INTERFACES, "::", "*", "", "192.168.1.10", "example.com", "10.0.0.1"],
)
def test_normalize_bind_host_refuses_non_loopback(host):
    with pytest.raises(BindError):
        normalize_bind_host(host)


def test_create_server_refuses_all_interfaces():
    with pytest.raises(BindError):
        create_server(host=ALL_INTERFACES, port=0)


def test_create_server_binds_loopback_only():
    httpd = create_server(host="127.0.0.1", port=0)
    try:
        host, port = httpd.server_address[:2]
        assert host == "127.0.0.1"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=2
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["status"] == "ok"
        assert body["bind"] == "127.0.0.1"
        assert body["weights"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_cli_refuses_all_interfaces(capsys):
    assert main(["--host", ALL_INTERFACES, "--port", "0"]) == 2
    assert "127.0.0.1" in capsys.readouterr().err


def test_repo_start_scripts_do_not_bind_all_interfaces():
    scripts = discover_start_scripts(REPO_ROOT)
    assert any(p.name == "start.sh" for p in scripts)
    assert_start_scripts_localhost_only(scripts)


def test_scan_fails_when_all_interfaces_in_start_script(tmp_path):
    script = tmp_path / "start.sh"
    script.write_text(f"python3 -m http.server --bind {ALL_INTERFACES}\n")
    with pytest.raises(AssertionError, match="not allowed in start scripts"):
        assert_start_scripts_localhost_only([script])
