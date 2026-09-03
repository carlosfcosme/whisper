import inspect

import whisper
from whisper.defaults import DEFAULT_DEVICE
from whisper.transcribe import cli


def test_default_device_is_cpu():
    assert DEFAULT_DEVICE == "cpu"
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_cli_device_argument_defaults_to_cpu():
    assert "default=DEFAULT_DEVICE" in inspect.getsource(cli)
