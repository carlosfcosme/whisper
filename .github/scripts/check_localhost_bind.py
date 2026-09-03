#!/usr/bin/env python3
"""Fail if bind is not 127.0.0.1-only.

Loads whisper/bind.py directly (no torch, no weight pull).
"""

from __future__ import print_function

import importlib.util
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BIND_PATH = os.path.join(REPO_ROOT, "whisper", "bind.py")


def _load_bind():
    spec = importlib.util.spec_from_file_location("whisper_bind", BIND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    if not os.path.isfile(BIND_PATH):
        print("missing whisper/bind.py", file=sys.stderr)
        return 1
    bind = _load_bind()
    if bind.BIND_HOST != "127.0.0.1":
        print("BIND_HOST must be 127.0.0.1", file=sys.stderr)
        return 1
    try:
        bind.require_bind_host("0.0.0.0")
    except bind.BindError:
        pass
    else:
        print("wildcard bind was accepted", file=sys.stderr)
        return 1
    sock = bind.bind_tcp(0)
    try:
        host, port = sock.getsockname()
        if host != "127.0.0.1" or port <= 0:
            print("bind_tcp did not listen on 127.0.0.1", file=sys.stderr)
            return 1
    finally:
        sock.close()
    print("OK: localhost-only bind")
    return 0


if __name__ == "__main__":
    sys.exit(main())
