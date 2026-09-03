"""Venv/offline tests: no model or network fetch; bind 127.0.0.1 only."""

import os
import socket
import subprocess
import venv
from pathlib import Path

import pytest

PROBE = Path(__file__).with_name("offline_venv_probe.py")


def _venv_python(root):
    if os.name == "nt":
        candidate = root / "Scripts" / "python.exe"
    else:
        candidate = root / "bin" / "python3"
        if not candidate.exists():
            candidate = root / "bin" / "python"
    return candidate


@pytest.fixture
def offline_venv(tmp_path):
    """Pip-free venv with system site-packages. Creating it does not use pip."""
    root = tmp_path / "offline-venv"
    builder = venv.EnvBuilder(
        with_pip=False, system_site_packages=True, clear=True, symlinks=True
    )
    builder.create(str(root))
    python = _venv_python(root)
    assert python.is_file(), "venv python missing at {}".format(python)
    assert not (root / "bin" / "pip").exists()
    assert not (root / "bin" / "pip3").exists()
    return root, python


def _offline_env(venv_root, cache_dir):
    env = os.environ.copy()
    env.update(
        {
            "VIRTUAL_ENV": str(venv_root),
            "WHISPER_CPU_ONLY": "1",
            "WHISPER_NO_WEIGHT_DOWNLOAD": "1",
            "WHISPER_LOCALHOST_ONLY": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "XDG_CACHE_HOME": str(cache_dir),
            "WHISPER_TEST_CACHE": str(cache_dir / "whisper"),
            # Blackhole proxy: any WAN fetch that escapes the guards fails locally.
            "http_proxy": "http://127.0.0.1:9",
            "https_proxy": "http://127.0.0.1:9",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "NO_PROXY": "127.0.0.1,localhost,::1",
            "no_proxy": "127.0.0.1,localhost,::1",
        }
    )
    return env


def test_offline_venv_prevents_fetch_and_requires_loopback_bind(offline_venv, tmp_path):
    root, python = offline_venv
    cache = tmp_path / "cache"
    cache.mkdir()
    result = subprocess.run(
        [str(python), str(PROBE), str(cache / "whisper")],
        env=_offline_env(root, cache),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        universal_newlines=True,
    )
    output = result.stdout + "\n" + result.stderr
    assert result.returncode == 0, output
    assert "in venv" in result.stdout
    assert "load_model(tiny) refused" in result.stdout
    assert "Hub download refused" in result.stdout
    assert "wildcard make_server refused" in result.stdout
    assert "serve bound 127.0.0.1" in result.stdout
    assert "no weight files written" in result.stdout
    assert list(cache.rglob("*.pt")) == []
    assert list(cache.rglob("*.safetensors")) == []


def test_offline_venv_python_is_isolated(offline_venv):
    root, python = offline_venv
    code = (
        "import sys, os; "
        "assert sys.prefix != sys.base_prefix; "
        "assert os.path.realpath(sys.prefix) == os.path.realpath({!r}); "
        "assert os.environ.get('VIRTUAL_ENV') == {!r}"
    ).format(str(root), str(root))
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(root)
    result = subprocess.run(
        [str(python), "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        universal_newlines=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_create_connection_to_hub_is_blocked_in_unit_tests():
    with pytest.raises(OSError, match="127.0.0.1"):
        socket.create_connection(("huggingface.co", 443), timeout=1)


def test_create_connection_loopback_still_works():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        client = socket.create_connection(("127.0.0.1", port), timeout=1)
        conn, addr = server.accept()
        try:
            assert addr[0] == "127.0.0.1"
        finally:
            conn.close()
            client.close()
    finally:
        server.close()


def test_probe_script_is_not_run_by_host_python_as_a_weight_pull(tmp_path):
    """The probe refuses named-model load; host python must do the same."""
    import whisper
    from whisper.runtime import WeightDownloadError

    cache = tmp_path / "host-cache"
    cache.mkdir()
    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny", download_root=str(cache))
    assert list(cache.iterdir()) == []
