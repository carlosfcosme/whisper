#!/usr/bin/env python3
"""Fail CI if sources bind all interfaces or a process listens off-loopback.

Scans ``whisper/``, ``.cursor/``, and ``scripts/`` for the all-interface
IPv4 bind token. Then loads bind/serve without importing torch and
asserts a 127.0.0.1 listen plus a refused all-interface bind.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import types
import urllib.request
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("whisper", ".cursor", "scripts")
TEXT_SUFFIXES = {".py", ".sh", ".json", ".md", ".yml", ".yaml", ".toml", ".txt"}
UNSPECIFIED = ".".join(("0",) * 4)
LOOPBACK = "127.0.0.1"


def find_unspecified_hits(root: Path | None = None) -> List[str]:
    hits = []
    scan_root = root if root is not None else ROOT
    dirs = SCAN_DIRS if root is None else ("whisper", ".cursor", "scripts")
    for dirname in dirs:
        base = scan_root / dirname
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix and path.suffix not in TEXT_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if UNSPECIFIED in text:
                hits.append(str(path.relative_to(scan_root)))
    return hits


def _load_submodules(*names: str) -> Dict[str, object]:
    """Load whisper submodules without executing whisper/__init__.py."""
    pkg_dir = ROOT / "whisper"
    existing = sys.modules.get("whisper")
    if existing is None or not getattr(existing, "__file__", None):
        pkg = types.ModuleType("whisper")
        pkg.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
        pkg.__package__ = "whisper"
        sys.modules["whisper"] = pkg
    loaded = {}
    for name in names:
        full = "whisper.%s" % name
        if full in sys.modules and getattr(sys.modules[full], "__file__", None):
            loaded[name] = sys.modules[full]
            continue
        path = pkg_dir / ("%s.py" % name)
        spec = importlib.util.spec_from_file_location(full, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        loaded[name] = mod
    return loaded


def runtime_checks() -> None:
    mods = _load_submodules("bind", "serve")
    bind = mods["bind"]
    serve = mods["serve"]

    for bad in (UNSPECIFIED, "::", "*", "", "8.8.8.8", "192.168.1.10", "::1"):
        try:
            bind.require_loopback_host(bad)
        except bind.BindError:
            continue
        raise SystemExit("bind accepted non-loopback host %r" % bad)

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
        with urllib.request.urlopen(
            "http://127.0.0.1:%s/health" % port, timeout=2
        ) as resp:
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


def main() -> int:
    unspecified = find_unspecified_hits()
    if unspecified:
        print("all-interface bind host is forbidden in application sources:")
        for hit in unspecified:
            print("  %s" % hit)
        return 1
    runtime_checks()
    print("ok: 127.0.0.1 bind; all-interface and non-loopback listen refused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
