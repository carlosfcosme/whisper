import whisper
from whisper.localhost import BIND_HOST


def test_default_device_is_cpu():
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_serve_policy_is_localhost():
    assert BIND_HOST == "127.0.0.1"
