#!/usr/bin/env python3
"""Fail unless bind_guard accepts 127.0.0.1 and refuses 0.0.0.0."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_bind_guard():
    path = Path(__file__).resolve().parents[1] / "whisper" / "bind_guard.py"
    spec = importlib.util.spec_from_file_location("whisper_bind_guard", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    guard = load_bind_guard()
    accepted = guard.bind_guard("127.0.0.1")
    if accepted != "127.0.0.1":
        sys.stderr.write("ERROR: bind_guard(127.0.0.1) returned %r\n" % (accepted,))
        return 1
    try:
        guard.bind_guard("0.0.0.0")
    except guard.BindError:
        pass
    else:
        sys.stderr.write("ERROR: bind_guard(0.0.0.0) must fail\n")
        return 1
    start = Path(__file__).resolve().parents[1] / ".cursor" / "start.sh"
    text = start.read_text()
    if "127.0.0.1" not in text or "0.0.0.0" in text:
        sys.stderr.write("ERROR: start.sh must bind 127.0.0.1 only\n")
        return 1
    sys.stdout.write("OK: bind_guard accepts 127.0.0.1 and fails on 0.0.0.0\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
