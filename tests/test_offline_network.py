import json
import socket
import threading
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from offline_guard import (
    NetworkDisabledError,
    guard_is_installed,
    is_loopback_host,
)

import whisper
from whisper.serve import ServeConfig, create_server


def test_network_guard_is_installed():
    assert guard_is_installed()


def test_loopback_hosts():
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("127.0.0.2")
    assert is_loopback_host("::1")
    assert is_loopback_host("localhost")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("8.8.8.8")
    assert not is_loopback_host("huggingface.co")
    assert not is_loopback_host("openaipublic.azureedge.net")


def test_guard_blocks_remote_dns_and_connect():
    with pytest.raises(NetworkDisabledError, match="loopback only"):
        socket.getaddrinfo("huggingface.co", 443)
    with pytest.raises(NetworkDisabledError, match="loopback only"):
        socket.create_connection(("1.1.1.1", 80), timeout=0.2)
    with pytest.raises((NetworkDisabledError, URLError, OSError)):
        urlopen("https://openaipublic.azureedge.net/", timeout=1)


def test_guard_blocks_wildcard_bind():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkDisabledError, match="bind"):
            sock.bind(("0.0.0.0", 0))
    finally:
        sock.close()


def test_guard_allows_loopback_bind_and_http():
    server = create_server(host="127.0.0.1", port=0, device="cpu")
    host, port = server.server_address
    assert host == "127.0.0.1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen("http://127.0.0.1:{0}/health".format(port)) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["host"] == "127.0.0.1"
        assert payload["device"] == "cpu"
        assert payload["hub"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_create_server_refuses_non_loopback_before_bind():
    with pytest.raises(ValueError, match="127.0.0.1"):
        create_server(ServeConfig(host="0.0.0.0", port=0))
    with pytest.raises(ValueError, match="127.0.0.1"):
        create_server(ServeConfig(host="8.8.8.8", port=0))


def test_named_checkpoint_does_not_download_with_network_disabled(tmp_path):
    with pytest.raises(RuntimeError, match="offline mode"):
        whisper.load_model("tiny", download_root=str(tmp_path), offline=True)
    assert list(tmp_path.iterdir()) == []


def test_ci_workflow_enforces_offline_runner():
    with open(".github/workflows/test.yml", encoding="utf-8") as handle:
        workflow = handle.read()
    with open("scripts/run_cpu_offline_tests.py", encoding="utf-8") as handle:
        runner = handle.read()
    assert "run_cpu_offline_tests.py" in workflow
    assert "WHISPER_TEST_DISABLE_NETWORK" in workflow
    assert "WHISPER_OFFLINE" in workflow
    assert "WHISPER_DEVICE: cpu" in workflow
    assert "not requires_cuda and not requires_weights" in runner
    assert "WHISPER_TEST_DISABLE_NETWORK" in runner
