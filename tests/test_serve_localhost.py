import importlib.util
import json
import subprocess
import sys
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
from whisper.sovereign import BIND_HOST

REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_INTERFACES = "0.0.0.0"


def discover_start_scripts(root: Path) -> List[Path]:
    """Return repo start scripts (start.sh / start-*.sh / environment start)."""
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
        text = path.read_text(encoding="utf-8")
        if ALL_INTERFACES in text:
            hits.append(str(path))
    return hits


def assert_start_scripts_localhost_only(paths: Iterable[Path]) -> None:
    hits = start_script_all_interface_hits(paths)
    assert hits == [], "{} is not allowed in start scripts: {}".format(
        ALL_INTERFACES, hits
    )


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "localhost", "LOCALHOST"],
)
def test_normalize_bind_host_allows_loopback(host):
    bound = normalize_bind_host(host)
    assert bound == "127.0.0.1"
    assert is_loopback_host(bound)


@pytest.mark.parametrize(
    "host",
    [ALL_INTERFACES, "::", "::1", "*", "", "192.168.1.10", "example.com", "10.0.0.1"],
)
def test_normalize_bind_host_refuses_non_loopback(host):
    with pytest.raises(BindError, match="127.0.0.1"):
        normalize_bind_host(host)


def test_create_server_refuses_ipv6_loopback():
    with pytest.raises(BindError, match="127.0.0.1"):
        create_server(host="::1", port=0)


def test_create_server_refuses_all_interfaces():
    with pytest.raises(BindError):
        create_server(host=ALL_INTERFACES, port=0)


def test_create_server_binds_loopback_only():
    httpd = create_server(host=BIND_HOST, port=0)
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
        assert body["hub"] is False
        assert body["weights"] is False
        assert body["device"] == "cpu"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_cli_refuses_all_interfaces(capsys):
    code = main(["--host", ALL_INTERFACES, "--port", "0"])
    assert code == 2
    err = capsys.readouterr().err
    assert "127.0.0.1" in err


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
    script.write_text("python3 -m http.server --bind {}\n".format(ALL_INTERFACES))
    with pytest.raises(AssertionError, match="not allowed in start scripts"):
        assert_start_scripts_localhost_only([script])


def _load_loopback_checker():
    path = REPO_ROOT / "scripts" / "check_loopback_bind.py"
    spec = importlib.util.spec_from_file_location("check_loopback_bind", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_loopback_ci_guard_passes_on_this_repo():
    script = REPO_ROOT / "scripts" / "check_loopback_bind.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "127.0.0.1" in result.stdout


def test_loopback_ci_guard_fails_on_all_interface_bind(tmp_path, monkeypatch):
    check = _load_loopback_checker()
    planted = tmp_path / "start.sh"
    planted.write_text("python3 -m http.server --bind {}\n".format(ALL_INTERFACES))
    monkeypatch.setattr(check, "tracked_files", lambda root: ["start.sh"])
    hits = check.find_hits(tmp_path)
    assert hits
    assert hits[0][0] == "start.sh"
    assert ALL_INTERFACES in hits[0][2]
