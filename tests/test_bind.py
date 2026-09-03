"""Loopback bind policy. No torch. No Hub. No weight download. No secrets."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ALL_INTERFACES = ".".join(("0", "0", "0", "0"))
LOOPBACK = "127.0.0.1"
CHECK_SCRIPT = REPO / "scripts" / "check_bind_localhost.py"


def _load_bind():
    path = REPO / "whisper" / "bind.py"
    spec = importlib.util.spec_from_file_location("whisper_bind_isolated", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bind = _load_bind()
BindError = bind.BindError
require_bind_127_0_0_1 = bind.require_bind_127_0_0_1
create_loopback_httpd = bind.create_loopback_httpd


class _OkHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


def test_ci_bind_script_passes():
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT)],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "bind-localhost: ok" in result.stdout


def test_empty_host_is_refused():
    with pytest.raises(BindError, match="required"):
        require_bind_127_0_0_1("")
    with pytest.raises(BindError, match="required"):
        require_bind_127_0_0_1("   ")


def test_all_interfaces_host_is_refused():
    with pytest.raises(BindError, match="all-interfaces"):
        require_bind_127_0_0_1(ALL_INTERFACES)
    with pytest.raises(BindError):
        require_bind_127_0_0_1("*")
    with pytest.raises(BindError):
        require_bind_127_0_0_1("::")


@pytest.mark.parametrize("host", ["8.8.8.8", "10.0.0.1", "example.com", "::1"])
def test_non_loopback_host_is_refused(host):
    with pytest.raises(BindError, match="127.0.0.1"):
        require_bind_127_0_0_1(host)


def test_localhost_rewrites_to_127_0_0_1():
    assert require_bind_127_0_0_1(None) == LOOPBACK
    assert require_bind_127_0_0_1("localhost") == LOOPBACK
    assert require_bind_127_0_0_1(LOOPBACK) == LOOPBACK


def test_application_sources_have_no_all_interfaces_token():
    hits = []
    for base in (REPO / "whisper", REPO / ".cursor"):
        for path in base.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if ALL_INTERFACES in text:
                hits.append(str(path.relative_to(REPO)))
    assert hits == [], "{} is not allowed in serve/listen paths: {}".format(
        ALL_INTERFACES, hits
    )


def test_start_sh_hardcodes_loopback():
    text = (REPO / ".cursor" / "start.sh").read_text(encoding="utf-8")
    assert "--host 127.0.0.1" in text
    assert ALL_INTERFACES not in text


def test_live_bind_is_127_0_0_1():
    httpd = create_loopback_httpd(_OkHandler, host=LOOPBACK, port=0)
    host, port = httpd.server_address[:2]
    assert host == LOOPBACK
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen("http://{}:{}/".format(LOOPBACK, port)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["status"] == "ok"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)
