#!/usr/bin/env python3
"""Fail CI if serve/bind defaults are not 127.0.0.1 or if 0.0.0.0 is allowed."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import threading
import types
from pathlib import Path
from urllib.request import urlopen

FORBIDDEN_DEFAULTS = ("0.0.0.0", "::", "*", "")


def _load_bind(path: Path):
    spec = importlib.util.spec_from_file_location("whisper_bind_check", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assigned_string_constants(path: Path, names: set[str]) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not (
            isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                values.append(node.value.value)
    return values


def _argparse_host_default(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_add = isinstance(func, ast.Attribute) and func.attr == "add_argument"
        if not is_add:
            continue
        if not any(
            isinstance(arg, ast.Constant) and arg.value == "--host" for arg in node.args
        ):
            continue
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Name):
                if kw.value.id != "LOOPBACK_HOST":
                    values.append(kw.value.id)
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                if kw.value.value != "127.0.0.1":
                    values.append(repr(kw.value.value))
    return values


def _load_serve(root: Path, bind):
    pkg = types.ModuleType("whisper")
    pkg.__path__ = [str(root / "whisper")]
    pkg.bind = bind
    sys.modules.setdefault("whisper", pkg)
    sys.modules["whisper.bind"] = bind
    spec = importlib.util.spec_from_file_location(
        "whisper.serve", root / "whisper" / "serve.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load whisper/serve.py")
    serve = importlib.util.module_from_spec(spec)
    sys.modules["whisper.serve"] = serve
    spec.loader.exec_module(serve)
    return serve


def _live_loopback(bind, serve) -> int:
    try:
        serve.create_server("0.0.0.0", 0)
    except bind.BindError:
        pass
    else:
        print("ERROR: create_server accepted 0.0.0.0", file=sys.stderr)
        return 1

    server = serve.create_server("127.0.0.1", 0)
    host, port = server.server_address[:2]
    if host != "127.0.0.1":
        print(f"ERROR: server bound {host!r}, not 127.0.0.1", file=sys.stderr)
        return 1
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    if payload.get("host") != "127.0.0.1" or payload.get("weights") is not False:
        print(f"ERROR: unexpected health payload {payload!r}", file=sys.stderr)
        return 1
    print(f"OK: live health on 127.0.0.1:{port}")
    return 0


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    bind_path = root / "whisper" / "bind.py"
    serve_path = root / "whisper" / "serve.py"
    bind = _load_bind(bind_path)

    if bind.LOOPBACK_HOST != "127.0.0.1":
        print("ERROR: LOOPBACK_HOST is not 127.0.0.1", file=sys.stderr)
        return 1

    for host in ("0.0.0.0", "::", "*", "", "192.168.1.10", "example.com"):
        try:
            bind.require_loopback(host)
        except bind.BindError:
            continue
        print(f"ERROR: require_loopback accepted {host!r}", file=sys.stderr)
        return 1

    if bind.require_loopback("localhost") != "127.0.0.1":
        print("ERROR: localhost must normalize to 127.0.0.1", file=sys.stderr)
        return 1
    if bind.require_loopback("127.0.0.1") != "127.0.0.1":
        print("ERROR: 127.0.0.1 must be accepted", file=sys.stderr)
        return 1

    for path in (bind_path, serve_path):
        for default in _assigned_string_constants(
            path, {"DEFAULT_HOST", "LOOPBACK_HOST"}
        ):
            if default in FORBIDDEN_DEFAULTS or default != "127.0.0.1":
                print(
                    f"ERROR: forbidden bind default {default!r} in {path}",
                    file=sys.stderr,
                )
                return 1

    bad_host_defaults = _argparse_host_default(serve_path)
    if bad_host_defaults:
        print(
            "ERROR: --host default is not LOOPBACK_HOST/127.0.0.1: "
            + ", ".join(bad_host_defaults),
            file=sys.stderr,
        )
        return 1

    start = root / ".cursor" / "start.sh"
    if start.is_file():
        text = start.read_text(encoding="utf-8")
        if "127.0.0.1" not in text:
            print("ERROR: .cursor/start.sh must bind 127.0.0.1", file=sys.stderr)
            return 1
        if "0.0.0.0" in text:
            print("ERROR: .cursor/start.sh mentions 0.0.0.0", file=sys.stderr)
            return 1

    serve = _load_serve(root, bind)
    live = _live_loopback(bind, serve)
    if live != 0:
        return live

    print("OK: bind policy is 127.0.0.1 only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
