import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Iterable, List

import pytest

from whisper.offline import BIND_HOST, DEFAULT_DEVICE
from whisper.serve import BindError, main, make_server, require_bind_127_0_0_1

REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_INTERFACES = "0.0.0.0"


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
        text = path.read_text(encoding="utf-8")
        if ALL_INTERFACES in text:
            hits.append(str(path))
    return hits


def assert_start_scripts_localhost_only(paths: Iterable[Path]) -> None:
    hits = start_script_all_interface_hits(paths)
    assert hits == [], "{} is not allowed in start scripts: {}".format(
        ALL_INTERFACES, hits
    )


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "LOCALHOST"])
def test_require_bind_allows_loopback(host):
    assert require_bind_127_0_0_1(host) == BIND_HOST


@pytest.mark.parametrize(
    "host",
    [
        ALL_INTERFACES,
        "::",
        "::1",
        "*",
        "",
        None,
        "192.168.1.10",
        "example.com",
        "10.0.0.1",
    ],
)
def test_require_bind_refuses_non_loopback(host):
    with pytest.raises(BindError, match="127.0.0.1"):
        require_bind_127_0_0_1(host)


def test_make_server_refuses_all_interfaces():
    with pytest.raises(BindError, match="127.0.0.1"):
        make_server(host=ALL_INTERFACES, port=0)


def test_make_server_binds_loopback_only():
    server = make_server(host=BIND_HOST, port=0)
    try:
        host, port = server.server_address[:2]
        assert host == BIND_HOST
        assert port > 0
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        with urllib.request.urlopen(
            "http://127.0.0.1:{}/health".format(port), timeout=2
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["ok"] is True
        assert body["bind"] == BIND_HOST
        assert body["hub"] is False
        assert body["weights"] is False
        assert body["device"] == DEFAULT_DEVICE
    finally:
        server.shutdown()
        server.server_close()


def test_cli_refuses_all_interfaces(capsys):
    code = main(["--host", ALL_INTERFACES, "--port", "0"])
    assert code == 2
    err = capsys.readouterr().err
    assert "127.0.0.1" in err


def test_start_script_exists_and_uses_loopback():
    start = REPO_ROOT / ".cursor" / "start.sh"
    assert start.is_file()
    text = start.read_text(encoding="utf-8")
    assert "--host 127.0.0.1" in text
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


def test_cli_subprocess_binds_127(tmp_path):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["XDG_CACHE_HOME"] = str(tmp_path)
    proc = subprocess.Popen(
        [sys.executable, "-m", "whisper.serve", "--host", BIND_HOST, "--port", "0"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        deadline = time.time() + 8
        bound = None
        buf = []
        while time.time() < deadline:
            if proc.poll() is not None:
                raise AssertionError(
                    "serve exited {}: {}{}".format(
                        proc.returncode, proc.stderr.read(), "".join(buf)
                    )
                )
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            buf.append(line)
            if "bound to 127.0.0.1:" in line:
                bound = line.strip()
                port = int(line.rsplit(":", 1)[1])
                break
        else:
            raise AssertionError("timed out waiting for bind: {}".format("".join(buf)))
        assert bound is not None
        with urllib.request.urlopen(
            "http://127.0.0.1:{}/".format(port), timeout=2
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["bind"] == BIND_HOST
        assert body["weights"] is False
        assert list(tmp_path.rglob("*.pt")) == []
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
