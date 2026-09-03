#!/usr/bin/env python3
"""Fail CI if application sources bind all interfaces or talk to the Hub.

Scans ``whisper/``, ``.cursor/``, and ``scripts/`` for the all-interface
IPv4 bind token and for Hugging Face Hub clients. Then loads bind/serve
without importing torch and asserts a 127.0.0.1 bind plus CPU default.
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
HUB_NEEDLES = (
    "hugging" + "face.co",
    "hugging" + "face_hub",
    "hf_" + "hub",
)
LOOPBACK = "127.0.0.1"


def _iter_scanned_files():
    for dirname in SCAN_DIRS:
        root = ROOT / dirname
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix and path.suffix not in TEXT_SUFFIXES:
                continue
            if path.name.endswith(".pyc"):
                continue
            yield path


def find_unspecified_hits() -> List[str]:
    hits = []
    for path in _iter_scanned_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if UNSPECIFIED in text:
            hits.append(str(path.relative_to(ROOT)))
    return hits


def find_hub_hits() -> List[str]:
    hits = []
    for path in _iter_scanned_files():
        try:
            text = path.read_text(encoding="utf-8").lower()
        except (UnicodeDecodeError, OSError):
            continue
        if any(needle in text for needle in HUB_NEEDLES):
            hits.append(str(path.relative_to(ROOT)))
    return hits


def _load_submodules(*names: str) -> Dict[str, object]:
    """Load whisper submodules without executing whisper/__init__.py (no torch)."""
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
    mods = _load_submodules("bind", "device", "serve")
    bind = mods["bind"]
    device = mods["device"]
    serve = mods["serve"]

    if device.default_device() != "cpu":
        raise SystemExit(
            "default_device() must be cpu, got %r" % device.default_device()
        )

    try:
        bind.require_loopback_host(UNSPECIFIED)
    except bind.BindError:
        pass
    else:
        raise SystemExit("bind accepted an all-interfaces host")

    httpd = serve.create_server(host=LOOPBACK, port=0)
    try:
        host, port = httpd.server_address[:2]
        if host != LOOPBACK:
            raise SystemExit("server bound %r, expected %s" % (host, LOOPBACK))
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        with urllib.request.urlopen(
            "http://127.0.0.1:%s/health" % port, timeout=2
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if body.get("bind") != LOOPBACK or body.get("hub") is not False:
            raise SystemExit("health payload failed: %s" % body)
        if body.get("device") != "cpu" or body.get("weights") is not False:
            raise SystemExit("health payload failed: %s" % body)
    finally:
        httpd.shutdown()
        httpd.server_close()


def main() -> int:
    unspecified = find_unspecified_hits()
    if unspecified:
        print("all-interface bind host is forbidden in application sources:")
        for hit in unspecified:
            print("  %s" % hit)
        return 1
    hub = find_hub_hits()
    if hub:
        print("Hugging Face Hub is forbidden in application sources:")
        for hit in hub:
            print("  %s" % hit)
        return 1
    runtime_checks()
    print("ok: 127.0.0.1 bind, CPU default, no Hub, no all-interface token")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
