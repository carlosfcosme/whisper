#!/usr/bin/env python3
"""Prove helper listeners bind 127.0.0.1 only.

Loads whisper/runtime.py without importing torch. Binds a real loopback
socket and refuses wildcard / public hosts.
"""

from __future__ import annotations

import importlib.util
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_HOSTS = ("0.0.0.0", "", "::", "8.8.8.8", "example.com")


def _load_runtime():
    spec = importlib.util.spec_from_file_location(
        "whisper_runtime", ROOT / "whisper" / "runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    runtime = _load_runtime()
    errors = []

    if runtime.default_bind_host() != "127.0.0.1":
        errors.append("default_bind_host() is {!r}".format(runtime.default_bind_host()))

    for host in FORBIDDEN_HOSTS:
        try:
            runtime.refuse_non_localhost_bind(host)
            errors.append("did not refuse bind host {!r}".format(host))
        except runtime.BindError:
            pass

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)
        host, port = runtime.bind_localhost(sock, 0)
        if host != "127.0.0.1" or port <= 0:
            errors.append("bind_localhost returned {!r}:{}".format(host, port))
        else:
            sock.listen(1)
            client = socket.create_connection(("127.0.0.1", port), timeout=1)
            try:
                conn, addr = sock.accept()
                try:
                    if addr[0] != "127.0.0.1":
                        errors.append("accepted peer {!r}".format(addr[0]))
                finally:
                    conn.close()
            finally:
                client.close()
    except OSError as exc:
        errors.append("loopback bind failed: {}".format(exc))
    finally:
        sock.close()

    if errors:
        sys.stderr.write("ERROR: bind is not loopback-only:\n")
        for item in errors:
            sys.stderr.write("  - {}\n".format(item))
        return 1

    sys.stdout.write("OK: listeners bind 127.0.0.1 only\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
