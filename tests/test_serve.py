"""Weights-free serve binds 127.0.0.1 and refuses all-interface hosts."""

from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from tests.import_stdlib import load_stdlib_modules

mods = load_stdlib_modules("bind", "serve")
bind = mods["bind"]
serve = mods["serve"]
LOOPBACK = bind.LOOPBACK_HOST
UNSPECIFIED = ".".join(("0",) * 4)


def test_create_server_binds_loopback():
    httpd = serve.create_server(host=LOOPBACK, port=0)
    try:
        host, port = httpd.server_address[:2]
        assert host == LOOPBACK
        bind.assert_no_nonloopback_listeners()
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        with urllib.request.urlopen(
            "http://127.0.0.1:%s/health" % port, timeout=2
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["status"] == "ok"
        assert body["bind"] == LOOPBACK
        assert body["weights"] is False
        assert body["hub"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.probes_bind
def test_create_server_refuses_all_interfaces():
    with pytest.raises(bind.BindError):
        serve.create_server(host=UNSPECIFIED, port=0)


@pytest.mark.probes_bind
def test_main_exits_2_on_all_interfaces():
    assert serve.main(["--host", UNSPECIFIED, "--port", "0"]) == 2
