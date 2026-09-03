import importlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from hub_guard import (
    FORBIDDEN_HUB_APIS,
    HubImportError,
    HubNetworkError,
    WanNetworkError,
    forbidden_hub_api_hits,
    is_hub_url,
    refuse_hub_url,
    refuse_wan_host,
    refuse_wan_url,
)
from offline_ci import (
    IGNORE_EXAMPLES,
    check_ci_has_no_default_weight_pull,
    check_gitignore,
    check_gitignore_examples,
    check_localhost_bind,
    check_no_hub_apis_in_tests,
    check_untracked_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_huggingface_hub_import_is_blocked():
    with pytest.raises((HubImportError, ImportError), match="huggingface_hub"):
        importlib.import_module("huggingface_hub")


def test_hf_hub_download_import_is_blocked():
    with pytest.raises((HubImportError, ImportError), match="huggingface_hub"):
        importlib.import_module("huggingface_hub.hf_hub_download")


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors",
        "https://cdn-lfs.huggingface.co/repos/tiny.pt",
        "https://hf.co/openai/whisper-tiny",
        "https://cas-bridge.xethub.hf.co/blob",
    ],
)
def test_hub_urls_are_refused(url):
    assert is_hub_url(url)
    with pytest.raises(HubNetworkError, match="Hugging Face Hub"):
        refuse_hub_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8765/health",
        "https://openaipublic.azureedge.net/main/whisper/models/abc/tiny.pt",
        "file:///tmp/jfk.flac",
    ],
)
def test_non_hub_urls_are_not_blocked_as_hub(url):
    assert not is_hub_url(url)
    refuse_hub_url(url)
    if url.startswith("http://127.0.0.1") or url.startswith("file:"):
        refuse_wan_url(url)


def test_urlopen_to_hub_is_blocked():
    import urllib.request

    with pytest.raises(HubNetworkError, match="Hugging Face Hub"):
        urllib.request.urlopen("https://huggingface.co/api/models")


@pytest.mark.parametrize(
    "url",
    [
        "https://openaipublic.azureedge.net/main/whisper/models/abc/tiny.pt",
        "https://example.com/tiny.pt",
        "http://8.8.8.8/tiny.pt",
        "http://10.0.0.1/tiny.pt",
    ],
)
def test_wan_urls_are_refused(url):
    with pytest.raises(WanNetworkError, match="WAN"):
        refuse_wan_url(url)


def test_urlopen_to_wan_is_blocked():
    import urllib.request

    with pytest.raises(WanNetworkError, match="WAN"):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/abc/tiny.pt"
        )


def test_socket_to_wan_is_blocked():
    import socket

    with pytest.raises(WanNetworkError, match="WAN"):
        socket.create_connection(("8.8.8.8", 53), timeout=0.2)
    with pytest.raises(WanNetworkError, match="WAN"):
        refuse_wan_host("openaipublic.azureedge.net")
    refuse_wan_host("127.0.0.1")


def test_cpu_is_the_default():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.environ.get("WHISPER_DEVICE") == "cpu"
    import torch

    from whisper.env_policy import resolve_device

    assert torch.cuda.is_available() is False
    assert resolve_device() == "cpu"
    assert resolve_device(None) == "cpu"


def test_no_weight_fetch_is_the_default():
    assert os.environ.get("WHISPER_ALLOW_WEIGHT_FETCH") == "0"
    from whisper.env_policy import weight_fetch_allowed

    assert weight_fetch_allowed() is False


def test_bind_host_defaults_to_loopback():
    assert os.environ.get("WHISPER_BIND_HOST") == "127.0.0.1"


def test_tests_do_not_reference_hub_download_apis():
    assert forbidden_hub_api_hits() == []
    check_no_hub_apis_in_tests()
    for name in FORBIDDEN_HUB_APIS:
        assert name


def test_committed_weights_and_caches_are_rejected():
    check_gitignore()
    check_untracked_weights()
    check_gitignore_examples()


def test_dummy_checkpoints_are_ignored_by_git():
    created = []
    try:
        for rel in IGNORE_EXAMPLES:
            path = REPO_ROOT / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not-a-real-checkpoint")
            created.append(path)
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *IGNORE_EXAMPLES],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert status.stdout.strip() == ""
    finally:
        for path in created:
            if path.exists():
                path.unlink()
            parent = path.parent
            while parent != REPO_ROOT and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent


def test_environment_and_ci_offline_policy():
    check_localhost_bind()
    check_ci_has_no_default_weight_pull()
    env = json.loads((REPO_ROOT / ".cursor/environment.json").read_text())
    assert "ports" not in env


def test_jfk_fixture_is_local():
    fixture = REPO_ROOT / "tests" / "jfk.flac"
    assert fixture.is_file()
    assert fixture.stat().st_size > 0
