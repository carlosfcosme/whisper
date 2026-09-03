import torch

import whisper
from whisper.device import DEFAULT_DEVICE, default_device
from whisper.model import ModelDimensions, Whisper
from whisper.transcribe import cli_parser


def _write_toy_checkpoint(path):
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=16,
        n_audio_state=32,
        n_audio_head=4,
        n_audio_layer=1,
        n_vocab=50,
        n_text_ctx=16,
        n_text_state=32,
        n_text_head=4,
        n_text_layer=1,
    )
    model = Whisper(dims)
    torch.save(
        {"dims": dims.__dict__, "model_state_dict": model.state_dict()},
        path,
    )


def test_default_device_is_cpu_when_cuda_available(monkeypatch):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert DEFAULT_DEVICE == "cpu"
    assert default_device() == "cpu"


def test_default_device_ignores_cuda_probe(monkeypatch):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)

    def boom():
        raise AssertionError("cuda.is_available must not choose the default")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    assert default_device() == "cpu"


def test_load_model_defaults_to_cpu(tmp_path, monkeypatch):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    ckpt = tmp_path / "toy.pt"
    _write_toy_checkpoint(ckpt)
    model = whisper.load_model(str(ckpt))
    assert model.device == torch.device("cpu")


def test_cli_device_default_is_cpu(monkeypatch):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    args = cli_parser().parse_args(["fixture.wav"])
    assert args.device == "cpu"
