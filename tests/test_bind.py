"""Loopback bind policy. Does not download model weights or use secrets."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_INTERFACES = "0.0.0.0"
LOOPBACK = "127.0.0.1"

# Paths whose contents may mention ALL_INTERFACES only as a rejected value
# (tests). Application serve/listen paths must not contain the token.
_SCAN_ROOTS = (
    REPO_ROOT / "whisper",
    REPO_ROOT / ".cursor",
)


def _load_bind_and_serve() -> Tuple[object, object]:
    try:
        import whisper.bind as bind_mod
        import whisper.serve as serve_mod

        return bind_mod, serve_mod
    except ImportError:
        pass

    bind_path = REPO_ROOT / "whisper" / "bind.py"
    serve_path = REPO_ROOT / "whisper" / "serve.py"
    bind_spec = importlib.util.spec_from_file_location(
        "whisper_bind_isolated", bind_path
    )
    bind_mod = importlib.util.module_from_spec(bind_spec)
    sys.modules["bind"] = bind_mod
    bind_spec.loader.exec_module(bind_mod)

    serve_spec = importlib.util.spec_from_file_location(
        "whisper_serve_isolated", serve_path
    )
    serve_mod = importlib.util.module_from_spec(serve_spec)
    serve_spec.loader.exec_module(serve_mod)
    return bind_mod, serve_mod


bind, serve = _load_bind_and_serve()
BindError = bind.BindError
require_loopback_bind = bind.require_loopback_bind
is_loopback_host = bind.is_loopback_host
create_server = serve.create_server
main = serve.main


def application_all_interface_hits(root: Optional[Path] = None) -> List[str]:
    """Return application paths that contain an all-interface bind token."""
    hits = []
    roots = (root,) if root is not None else _SCAN_ROOTS
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if ".git" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if ALL_INTERFACES in text:
                hits.append(str(path.relative_to(REPO_ROOT) if root is None else path))
    return hits


def assert_application_sources_localhost_only(
    roots: Optional[Iterable[Path]] = None,
) -> None:
    if roots is None:
        hits = application_all_interface_hits()
    else:
        hits = []
        for root in roots:
            hits.extend(application_all_interface_hits(root))
    assert hits == [], f"{ALL_INTERFACES} is not allowed in serve/listen paths: {hits}"


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


@pytest.mark.parametrize("host", [LOOPBACK, "localhost", "LOCALHOST"])
def test_require_loopback_bind_allows_loopback(host):
    bound = require_loopback_bind(host)
    assert bound == LOOPBACK
    assert is_loopback_host(bound)


@pytest.mark.parametrize(
    "host",
    [
        ALL_INTERFACES,
        "::",
        "*",
        "",
        "192.168.1.10",
        "example.com",
        "10.0.0.1",
        "8.8.8.8",
        "172.16.0.1",
        "[::]",
    ],
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
    httpd = create_server(host=LOOPBACK, port=0)
    try:
        host, port = httpd.server_address[:2]
        assert host == LOOPBACK
        assert is_loopback_host(host)
        assert httpd.socket.getsockname()[0] == LOOPBACK
        assert port > 0
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=2
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["status"] == "ok"
        assert body["bind"] == LOOPBACK
        assert body["weights"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_live_listen_is_not_all_interfaces():
    httpd = create_server(host=LOOPBACK, port=0)
    try:
        port = httpd.server_address[1]
        assert httpd.socket.getsockname()[0] == LOOPBACK
        bind.assert_loopback_socket(httpd.socket, require_proc=True)
        hosts = bind.observed_listen_hosts(port)
        assert hosts, "CI must observe the listen in /proc"
        assert ALL_INTERFACES not in hosts
        assert "::" not in hosts
        assert LOOPBACK in hosts
        assert hosts == [LOOPBACK] or set(hosts) == {LOOPBACK}
    finally:
        httpd.server_close()


def test_cli_refuses_all_interfaces(capsys):
    code = main(["--host", ALL_INTERFACES, "--port", "0"])
    assert code == 2
    err = capsys.readouterr().err
    assert "127.0.0.1" in err


@pytest.mark.parametrize("host", [ALL_INTERFACES, "192.168.1.10", "8.8.8.8", "::"])
def test_cli_refuses_non_loopback(host, capsys):
    code = main(["--host", host, "--port", "0"])
    assert code == 2
    assert "127.0.0.1" in capsys.readouterr().err


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
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        assert ALL_INTERFACES not in text, path


def test_application_sources_do_not_contain_all_interfaces():
    assert_application_sources_localhost_only()


def test_scan_fails_when_all_interfaces_in_start_script(tmp_path):
    script = tmp_path / "start.sh"
    script.write_text(f"python3 -m http.server --bind {ALL_INTERFACES}\n")
    with pytest.raises(AssertionError, match="not allowed in serve/listen paths"):
        assert_application_sources_localhost_only([tmp_path])


def test_weight_paths_are_gitignored():
    for path in (
        "tiny.pt",
        "weights/tiny.pt",
        ".cache/whisper/tiny.pt",
        "model.pth",
        ".env",
        ".env.local",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", path],
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, path


def test_no_weight_download_in_serve_path():
    serve_src = (REPO_ROOT / "whisper" / "serve.py").read_text(encoding="utf-8")
    bind_src = (REPO_ROOT / "whisper" / "bind.py").read_text(encoding="utf-8")
    for src in (serve_src, bind_src):
        assert "load_model" not in src
        assert "_download" not in src
        assert "azureedge.net" not in src


def test_check_loopback_listen_script_passes():
    spec = importlib.util.spec_from_file_location(
        "check_loopback_listen",
        REPO_ROOT / "scripts" / "check_loopback_listen.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == 0
