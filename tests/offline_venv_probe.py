#!/usr/bin/env python3
"""Run inside a pip-free venv: no model/network fetch, bind 127.0.0.1 only.

This script is invoked by tests/test_venv_offline.py. It must not call pip,
load_model successfully, or bind a wildcard address.
"""

from __future__ import print_function

import os
import socket
import sys
import threading
from urllib.parse import urlparse
from urllib.request import urlopen


def _loopback(host):
    return (host or "").lower() in {"127.0.0.1", "localhost", "::1"}


def _install_network_guards():
    import urllib.request

    real_urlopen = urllib.request.urlopen

    def guarded_urlopen(url, *args, **kwargs):
        target = url if isinstance(url, str) else getattr(url, "full_url", str(url))
        host = (urlparse(target).hostname or "").lower()
        if not _loopback(host):
            raise RuntimeError("offline venv blocked remote URL: {}".format(target))
        return real_urlopen(url, *args, **kwargs)

    urllib.request.urlopen = guarded_urlopen

    real_cc = socket.create_connection

    def guarded_cc(address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if not _loopback(str(host)):
            raise OSError("offline venv blocked connect to {!r}".format(address))
        return real_cc(address, *args, **kwargs)

    socket.create_connection = guarded_cc


def _fail(msg):
    print("FAIL: {}".format(msg), file=sys.stderr)
    sys.exit(1)


def _ok(msg):
    print("OK: {}".format(msg))


def main(argv):
    cache_dir = argv[1] if len(argv) > 1 else os.environ.get("WHISPER_TEST_CACHE")
    if not cache_dir:
        _fail("cache dir argument required")
    os.makedirs(cache_dir, exist_ok=True)

    if sys.prefix == sys.base_prefix:
        _fail("probe must run inside a venv (prefix == base_prefix)")
    _ok("in venv prefix={}".format(sys.prefix))

    if os.path.exists(os.path.join(sys.prefix, "bin", "pip")) or os.path.exists(
        os.path.join(sys.prefix, "bin", "pip3")
    ):
        _fail("venv must be pip-free (no network install)")
    _ok("venv has no pip")

    _install_network_guards()

    import whisper
    from whisper.runtime import BindError, WeightDownloadError, default_bind_host
    from whisper.serve import make_server, serve_cli

    if whisper.default_device() != "cpu":
        _fail("default_device is {!r}, expected cpu".format(whisper.default_device()))
    _ok("default_device=cpu")

    if default_bind_host() != "127.0.0.1":
        _fail("default_bind_host is {!r}".format(default_bind_host()))
    _ok("default_bind_host=127.0.0.1")

    hub = "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"
    try:
        whisper._download(hub, cache_dir, in_memory=False)
    except WeightDownloadError as exc:
        if "Hugging Face Hub" not in str(exc):
            _fail("Hub refuse message missing: {!r}".format(exc))
        _ok("Hub download refused")
    else:
        _fail("Hub _download must not succeed")

    try:
        whisper.load_model("tiny", download_root=cache_dir)
    except WeightDownloadError:
        _ok("load_model(tiny) refused (no model fetch)")
    else:
        _fail("load_model(tiny) must not fetch or load weights")

    try:
        urlopen(hub, timeout=1)
    except Exception:
        _ok("urlopen Hub blocked")
    else:
        _fail("urlopen Hub must not succeed")

    try:
        socket.create_connection(("huggingface.co", 443), timeout=1)
    except OSError:
        _ok("create_connection Hub blocked")
    else:
        _fail("create_connection to huggingface.co must not succeed")

    try:
        make_server("0.0.0.0", 0)
    except BindError:
        _ok("wildcard make_server refused")
    else:
        _fail("make_server(0.0.0.0) must raise BindError")

    if serve_cli(["--host", "0.0.0.0", "--port", "0"]) != 2:
        _fail("serve_cli wildcard host must return 2")
    _ok("serve_cli wildcard rejected")

    httpd = make_server("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.handle_request)
    thread.daemon = True
    thread.start()
    host, port = httpd.server_address[:2]
    if host != "127.0.0.1":
        _fail("server bound {!r}, expected 127.0.0.1".format(host))
    try:
        with urlopen("http://127.0.0.1:{}/".format(port), timeout=2) as resp:
            body = resp.read()
    finally:
        thread.join(2)
        httpd.server_close()
    if body != b"ok\n":
        _fail("loopback health body={!r}".format(body))
    _ok("serve bound 127.0.0.1 and answered locally")

    leftover = []
    for root, _dirs, files in os.walk(cache_dir):
        for name in files:
            if name.endswith((".pt", ".bin", ".safetensors", ".pth")):
                leftover.append(os.path.join(root, name))
    if leftover:
        _fail("weight files written: {}".format(leftover))
    _ok("no weight files written")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
