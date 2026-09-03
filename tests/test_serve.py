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
    bind_mod = None
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
        bind_mod = loaded.get("bind")
        yield loaded
    finally:
        if bind_mod is not None and hasattr(bind_mod, "uninstall_bind_guard"):
            bind_mod.uninstall_bind_guard()
        for key in list(sys.modules):
            if key == "whisper" or key.startswith("whisper."):
                del sys.modules[key]
        sys.modules.update(saved)


def test_create_server_refuses_all_interfaces():
    with _isolated_whisper_modules("bind", "serve") as mods:
        with pytest.raises(mods["bind"].BindError):
            mods["serve"].create_server(host=ALL_INTERFACES, port=0)


def test_server_never_binds_all_interfaces():
    """Regression: a live server must not listen on the IPv4 wildcard."""
    with _isolated_whisper_modules("bind", "serve") as mods:
        bind = mods["bind"]
        serve = mods["serve"]
        with pytest.raises(bind.BindError, match="127.0.0.1"):
            serve.create_server(host=ALL_INTERFACES, port=0)
        bind.install_bind_guard()
        raw = bind.socket.socket(bind.socket.AF_INET, bind.socket.SOCK_STREAM)
        try:
            with pytest.raises(bind.BindError, match="127.0.0.1"):
                raw.bind((ALL_INTERFACES, 0))
        finally:
            raw.close()
        httpd = serve.create_server(host=LOOPBACK, port=0)
        try:
            host, port = httpd.server_address[:2]
            assert host == LOOPBACK
            assert host != ALL_INTERFACES
            assert port > 0
            bind.assert_no_nonloopback_listeners()
            for row in bind.process_listen_records():
                assert row.host == LOOPBACK
                assert row.host != ALL_INTERFACES
        finally:
            httpd.shutdown()
            httpd.server_close()
            bind.uninstall_bind_guard()


@pytest.mark.parametrize("host", ["::", "*", "", "10.0.0.1", "example.com"])
def test_create_server_refuses_non_loopback(host):
    with _isolated_whisper_modules("bind", "serve") as mods:
        with pytest.raises(mods["bind"].BindError):
            mods["serve"].create_server(host=host, port=0)


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
            mods["bind"].uninstall_bind_guard()


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


def test_cli_dispatches_serve():
    source = (ROOT / "whisper" / "transcribe.py").read_text(encoding="utf-8")
    assert 'sys.argv[1] == "serve"' in source
    assert "serve_main" in source


def test_serve_module_is_weights_free():
    source = (ROOT / "whisper" / "serve.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "load_model(" not in source
    assert "huggingface" not in source
    assert ALL_INTERFACES not in source
