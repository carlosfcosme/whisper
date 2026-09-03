#!/usr/bin/env python3
"""CI check: this process must not LISTEN off 127.0.0.1.

Loads whisper/bind.py by path so the job needs no torch, no weights, and no
credentials. Offline after checkout. No kernel modules, no BPF.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIND_PATH = ROOT / "whisper" / "bind.py"
WHISPER_DIR = ROOT / "whisper"


def load_bind():
    spec = importlib.util.spec_from_file_location("whisper_bind_guard", BIND_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise SystemExit(f"cannot load {BIND_PATH}")
    spec.loader.exec_module(module)
    return module


def main() -> int:
    bind = load_bind()
    try:
        bind.assert_only_loopback_listeners()
    except bind.NonLoopbackListenError as exc:
        print(exc, file=sys.stderr)
        return 1

    literals = bind.find_wildcard_host_literals(WHISPER_DIR)
    if literals:
        print("wildcard host literal(s) in whisper/:", file=sys.stderr)
        for path in literals:
            print(f"  {path}", file=sys.stderr)
        return 1

    calls = bind.find_non_loopback_bind_calls(WHISPER_DIR)
    if calls:
        print("non-loopback bind/listen call(s):", file=sys.stderr)
        for item in calls:
            print(f"  {item}", file=sys.stderr)
        return 1

    print("ok: no non-loopback LISTEN; whisper/ binds 127.0.0.1 only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
