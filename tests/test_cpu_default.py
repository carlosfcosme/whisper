import torch

import whisper
from whisper.model import ModelDimensions, Whisper
from whisper.runtime import DEFAULT_DEVICE, default_device


def _write_toy_checkpoint(path):
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=16,
        n_audio_state=16,
        n_audio_head=2,
        n_audio_layer=1,
        n_vocab=100,
        n_text_ctx=16,
        n_text_state=16,
        n_text_head=2,
        n_text_layer=1,
    )
    model = Whisper(dims)
    torch.save(
        {"dims": dims.__dict__, "model_state_dict": model.state_dict()},
        path,
    )


def test_default_device_is_cpu_not_cuda(monkeypatch):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert DEFAULT_DEVICE == "cpu"
    assert default_device() == "cpu"
    assert whisper.default_device() == "cpu"
    assert default_device() != "cuda"
    assert torch.device(default_device()).type == "cpu"


def test_load_model_local_checkpoint_lands_on_cpu(tmp_path, monkeypatch):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    ckpt = tmp_path / "toy.pt"
    _write_toy_checkpoint(ckpt)
    model = whisper.load_model(str(ckpt))
    assert model.device.type == "cpu"
