#!/usr/bin/env python3
"""Fail CI if sources bind all interfaces or a process listens off-loopback.

Scans ``whisper/``, ``.cursor/``, ``scripts/``, and ``.github/`` for the
IPv4 all-interfaces token. Then loads bind/serve without importing
torch and asserts a live ``127.0.0.1`` listen plus a refused
all-interface host. This process's listen sockets must stay on
``127.0.0.1``.

No weight download, no credentials, no WAN, no external service
activation.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import types
from pathlib import Path
from typing import Dict, List
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("whisper", ".cursor", "scripts", ".github")
TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".json",
    ".md",
    ".yml",
    ".yaml",
    ".toml",
    ".txt",
}
UNSPECIFIED = ".".join(("0",) * 4)
LOOPBACK = "127.0.0.1"


def _iter_scanned_files(root: Path = ROOT):
    for dirname in SCAN_DIRS:
        base = root / dirname
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix and path.suffix not in TEXT_SUFFIXES:
                continue
            if path.name.endswith(".pyc"):
                continue
            yield path


def find_unspecified_hits(root: Path = ROOT) -> List[str]:
    hits = []
    for path in _iter_scanned_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if UNSPECIFIED in text:
            hits.append(str(path.relative_to(root)))
    return hits


def _load_submodules(*names: str) -> Dict[str, object]:
    """Load whisper submodules without executing whisper/__init__.py."""
    pkg_dir = ROOT / "whisper"
    pkg = types.ModuleType("whisper")
    pkg.__path__ = [str(pkg_dir)]
    pkg.__package__ = "whisper"
    sys.modules["whisper"] = pkg
    loaded = {}
    for name in names:
        path = pkg_dir / ("%s.py" % name)
        spec = importlib.util.spec_from_file_location("whisper.%s" % name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["whisper.%s" % name] = mod
        spec.loader.exec_module(mod)
        loaded[name] = mod
    return loaded


def runtime_checks() -> None:
    mods = _load_submodules("bind", "serve")
    bind = mods["bind"]
    serve = mods["serve"]

    if bind.LOOPBACK_HOST != LOOPBACK:
        raise SystemExit("LOOPBACK_HOST must be %s" % LOOPBACK)
    if bind.UNSPECIFIED_V4 != UNSPECIFIED:
        raise SystemExit("UNSPECIFIED_V4 mismatch")

    try:
        bind.require_loopback_host(UNSPECIFIED)
    except bind.BindError:
        pass
    else:
        raise SystemExit("bind accepted an all-interfaces host")

    for host in (
        UNSPECIFIED,
        "::",
        "*",
        "",
        "   ",
        "8.8.8.8",
        "10.0.0.1",
        "192.168.1.10",
        "::1",
        "127.0.0.2",
    ):
        try:
            bind.require_loopback_host(host)
        except bind.BindError:
            continue
        raise SystemExit("bind accepted non-loopback host %r" % host)

    if bind.require_loopback_host(None) != LOOPBACK:
        raise SystemExit("default host is not %s" % LOOPBACK)
    if bind.require_loopback_host("localhost") != LOOPBACK:
        raise SystemExit("localhost was not rewritten to %s" % LOOPBACK)

    bind.install_bind_guard()
    httpd = serve.create_server(host=LOOPBACK, port=0)
    try:
        host, port = httpd.server_address[:2]
        if host != LOOPBACK:
            raise SystemExit("server bound %r, expected %s" % (host, LOOPBACK))
        bind.assert_no_nonloopback_listeners()
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        bind.assert_no_nonloopback_listeners()
        with urlopen("http://127.0.0.1:%s/health" % port, timeout=2) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if body.get("bind") != LOOPBACK or body.get("status") != "ok":
            raise SystemExit("health payload failed: %s" % body)
        if body.get("weights") is not False:
            raise SystemExit("health payload failed: %s" % body)
        try:
            serve.create_server(host=UNSPECIFIED, port=0)
        except bind.BindError:
            pass
        else:
            raise SystemExit("serve accepted an all-interfaces host")
    finally:
        httpd.shutdown()
        httpd.server_close()
        bind.uninstall_bind_guard()

    start = ROOT / ".cursor" / "start.sh"
    if not start.is_file():
        raise SystemExit("missing .cursor/start.sh")
    start_text = start.read_text(encoding="utf-8")
    if LOOPBACK not in start_text:
        raise SystemExit(".cursor/start.sh must bind %s" % LOOPBACK)
    if UNSPECIFIED in start_text:
        raise SystemExit(".cursor/start.sh must not mention all-interfaces")
    if "whisper.serve" not in start_text:
        raise SystemExit(".cursor/start.sh must launch whisper.serve")
    if "--live" in start_text:
        raise SystemExit(".cursor/start.sh must not pass --live")


def planted_unspecified_is_detected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        planted = root / "whisper"
        planted.mkdir()
        (planted / "bad.py").write_text("host = %r\n" % UNSPECIFIED)
        hits = find_unspecified_hits(root)
        if "whisper/bad.py" not in [Path(h).as_posix() for h in hits]:
            raise SystemExit("checker missed a planted all-interfaces token")


def main() -> int:
    unspecified = find_unspecified_hits()
    if unspecified:
        print("all-interface bind host is forbidden in application sources:")
        for hit in unspecified:
            print("  %s" % hit)
        return 1
    planted_unspecified_is_detected()
    runtime_checks()
    print("ok: 127.0.0.1 bind; all-interface and non-loopback listen refused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
