import importlib.util
import json
import sys
import threading
import types
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ALL_INTERFACES = ".".join(("0",) * 4)
LOOPBACK = "127.0.0.1"


@contextmanager
def _isolated_whisper_modules(*names):
    saved = {
        key: sys.modules[key]
        for key in list(sys.modules)
        if key == "whisper" or key.startswith("whisper.")
    }
    for key in saved:
        del sys.modules[key]
    try:
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
        yield loaded
    finally:
        for key in list(sys.modules):
            if key == "whisper" or key.startswith("whisper."):
                del sys.modules[key]
        sys.modules.update(saved)


def test_create_server_refuses_all_interfaces():
    with _isolated_whisper_modules("bind", "serve") as mods:
        with pytest.raises(mods["bind"].BindError):
            mods["serve"].create_server(host=ALL_INTERFACES, port=0)


def test_create_server_binds_loopback_only():
    with _isolated_whisper_modules("bind", "serve") as mods:
        httpd = mods["serve"].create_server(host=LOOPBACK, port=0)
        try:
            host, port = httpd.server_address[:2]
            assert host == LOOPBACK
            assert port > 0
            mods["bind"].assert_no_nonloopback_listeners()
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            mods["bind"].assert_no_nonloopback_listeners()
            url = "http://127.0.0.1:%s/health" % port
            with urllib.request.urlopen(url, timeout=2) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            assert body["status"] == "ok"
            assert body["bind"] == LOOPBACK
            assert body["weights"] is False
        finally:
            httpd.shutdown()
            httpd.server_close()


def test_cli_refuses_all_interfaces(capsys):
    with _isolated_whisper_modules("bind", "serve") as mods:
        code = mods["serve"].main(["--host", ALL_INTERFACES, "--port", "0"])
    assert code == 2
    err = capsys.readouterr().err
    assert "127.0.0.1" in err


def test_cli_refuses_live_flag(capsys):
    with _isolated_whisper_modules("bind", "serve") as mods:
        code = mods["serve"].main(["--live"])
    assert code == 2
    assert "refused" in capsys.readouterr().err


def test_start_script_binds_loopback_only():
    start = ROOT / ".cursor" / "start.sh"
    text = start.read_text(encoding="utf-8")
    assert "127.0.0.1" in text
    assert "whisper.serve" in text
    assert ALL_INTERFACES not in text
    assert "--live" not in text
