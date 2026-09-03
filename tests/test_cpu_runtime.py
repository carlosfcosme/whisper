"""Runtime CPU default. Requires the installed package (torch)."""

import inspect

import whisper


def test_default_device_constant_is_cpu():
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_load_model_source_defaults_to_cpu():
    source = inspect.getsource(whisper.load_model)
    assert "device = DEFAULT_DEVICE" in source
    assert "cuda.is_available" not in source
