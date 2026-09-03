import os
import urllib.request

import pytest
from hub_offline import (
    HUB_OFFLINE_ENV,
    is_huggingface_hub_host,
    refuse_hub_download,
    urlopen_without_hub,
)


@pytest.mark.parametrize(
    "host",
    [
        "huggingface.co",
        "www.huggingface.co",
        "hf.co",
        "cdn-lfs.huggingface.co",
        "cas-bridge.xethub.hf.co",
    ],
)
def test_huggingface_hosts_are_blocked(host):
    assert is_huggingface_hub_host(host)


def test_loopback_is_not_a_hub_host():
    assert not is_huggingface_hub_host("127.0.0.1")
    assert not is_huggingface_hub_host("localhost")


def test_hub_offline_env_is_set():
    for name in HUB_OFFLINE_ENV:
        assert os.environ.get(name) == "1"


def test_urlopen_refuses_huggingface_hub():
    with pytest.raises(RuntimeError, match="Hugging Face Hub"):
        urlopen_without_hub("https://huggingface.co/openai/whisper-tiny")


def test_hub_client_guard_raises():
    with pytest.raises(RuntimeError, match="Hugging Face Hub"):
        refuse_hub_download("openai/whisper-tiny", "model.safetensors")


def test_urlopen_wrapper_is_installed():
    assert urllib.request.urlopen is urlopen_without_hub
