#!/usr/bin/env python3
"""Fail CI if package code binds off-box. Enforce 127.0.0.1."""

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIND_HOST = "127.0.0.1"
WILDCARD = "0.0.0.0"


def main() -> int:
    listed = subprocess.run(
        ["git", "grep", "-nF", WILDCARD, "--", "whisper"],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if listed.returncode == 0 and listed.stdout.strip():
        print("Package code must bind %s only:" % BIND_HOST, file=sys.stderr)
        print(listed.stdout, file=sys.stderr)
        return 1
    if listed.returncode not in (0, 1):
        print(listed.stderr, file=sys.stderr)
        return 1

    spec = importlib.util.spec_from_file_location(
        "whisper_bind", ROOT / "whisper" / "bind.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.BIND_HOST != BIND_HOST:
        print("BIND_HOST must be %s" % BIND_HOST, file=sys.stderr)
        return 1
    if module.require_bind_127_0_0_1(BIND_HOST) != BIND_HOST:
        print("require_bind_127_0_0_1(%r) failed" % BIND_HOST, file=sys.stderr)
        return 1
    for host in (None, "", "localhost", WILDCARD, "192.168.1.1"):
        try:
            module.require_bind_127_0_0_1(host)
        except module.BindError:
            continue
        print("require_bind_127_0_0_1 accepted %r" % (host,), file=sys.stderr)
        return 1

    print("ok: serve/bind is %s only" % BIND_HOST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
