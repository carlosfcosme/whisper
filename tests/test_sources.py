import pytest

import whisper
from whisper.sources import (
    DEFAULT_BIND_HOST,
    DEFAULT_DEVICE,
    default_device,
    is_hub_source,
    reject_hub_source,
    require_loopback_host,
)


def test_default_device_is_cpu():
    assert DEFAULT_DEVICE == "cpu"
    assert default_device() == "cpu"


def test_load_model_rejects_hub_urls():
    with pytest.raises(ValueError, match="Hub"):
        whisper.load_model("https://huggingface.co/openai/whisper-tiny")
    with pytest.raises(ValueError, match="Hub"):
        whisper.load_model("hf://openai/whisper-tiny")


@pytest.mark.parametrize(
    "source,expected",
    [
        ("tiny", False),
        ("tiny.en", False),
        ("/tmp/local.pt", False),
        ("./checkpoints/base.pt", False),
        ("https://huggingface.co/openai/whisper-tiny", True),
        ("https://hf.co/openai/whisper-tiny", True),
        ("hf://openai/whisper-tiny", True),
        ("https://hf-mirror.com/openai/whisper-tiny", True),
        (None, False),
    ],
)
def test_is_hub_source(source, expected):
    assert is_hub_source(source) is expected


def test_reject_hub_source_passthrough():
    reject_hub_source("tiny")
    reject_hub_source("/models/tiny.pt")


def test_require_loopback_host():
    assert require_loopback_host("127.0.0.1") == DEFAULT_BIND_HOST
    with pytest.raises(ValueError, match="127.0.0.1"):
        require_loopback_host("0.0.0.0")
    with pytest.raises(ValueError, match="127.0.0.1"):
        require_loopback_host("localhost")
