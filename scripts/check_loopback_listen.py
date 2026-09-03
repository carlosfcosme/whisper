#!/usr/bin/env python3
"""Fail CI if serve/listen accepts or observes a non-loopback bind.

Offline-safe: no WAN, no model weights, no secrets. Loads whisper.bind /
whisper.serve by file path so torch is not required.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relpath: str, extra_sys_modules=None):
    if extra_sys_modules:
        sys.modules.update(extra_sys_modules)
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_bind_and_serve():
    bind_mod = _load("whisper_bind_ci", "whisper/bind.py")
    sys.modules["bind"] = bind_mod
    serve_mod = _load("whisper_serve_ci", "whisper/serve.py")
    return bind_mod, serve_mod


def refuse_non_loopback(bind_mod, serve_mod) -> List[str]:
    failures = []
    hosts = [
        bind_mod.ALL_INTERFACES,
        "::",
        "*",
        "",
        "192.168.1.10",
        "10.0.0.1",
        "8.8.8.8",
        "172.16.0.1",
        "example.com",
    ]
    for host in hosts:
        try:
            bind_mod.require_loopback_bind(host)
            failures.append("policy accepted non-loopback host {!r}".format(host))
        except bind_mod.BindError:
            pass
        try:
            httpd = serve_mod.create_server(host=host, port=0)
            httpd.server_close()
            failures.append("create_server bound non-loopback host {!r}".format(host))
        except bind_mod.BindError:
            pass
    code = serve_mod.main(["--host", bind_mod.ALL_INTERFACES, "--port", "0"])
    if code != 2:
        failures.append("CLI --host all-interfaces exited {} (want 2)".format(code))
    return failures


def prove_loopback_listen(bind_mod, serve_mod) -> List[str]:
    failures = []
    httpd = serve_mod.create_server(host=bind_mod.LOOPBACK_HOST, port=0)
    try:
        host, port = httpd.server_address[:2]
        if host != bind_mod.LOOPBACK_HOST:
            failures.append("server_address is {!r}".format(host))
        sock_host = httpd.socket.getsockname()[0]
        if sock_host != bind_mod.LOOPBACK_HOST:
            failures.append("getsockname is {!r}".format(sock_host))
        try:
            bind_mod.assert_loopback_socket(httpd.socket, require_proc=True)
        except bind_mod.BindError as exc:
            failures.append("assert_loopback_socket: {}".format(exc))
        observed = bind_mod.observed_listen_hosts(port)
        if bind_mod.ALL_INTERFACES in observed or "::" in observed:
            failures.append("non-loopback listen observed: {}".format(observed))
        if bind_mod.LOOPBACK_HOST not in observed:
            failures.append("127.0.0.1 listen missing; observed {}".format(observed))
    finally:
        httpd.server_close()
    return failures


def main() -> int:
    bind_mod, serve_mod = _load_bind_and_serve()
    failures = refuse_non_loopback(bind_mod, serve_mod)
    failures.extend(prove_loopback_listen(bind_mod, serve_mod))
    if failures:
        sys.stderr.write("ERROR: non-loopback listen/bind is not allowed:\n")
        for item in failures:
            sys.stderr.write("  {}\n".format(item))
        return 1
    sys.stdout.write("OK: listen is 127.0.0.1 only; non-loopback hosts refused\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
