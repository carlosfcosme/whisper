import os
import random as rand
from pathlib import Path

import numpy
import pytest
import torch

from whisper.model import ModelDimensions, Whisper

_OFFLINE_VALUES = frozenset({"1", "true", "yes", "on"})


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line(
        "markers", "requires_weights: tests that load official model checkpoints"
    )


def pytest_collection_modifyitems(config, items):
    flag = os.getenv("WHISPER_OFFLINE", "1").lower()
    if flag not in _OFFLINE_VALUES:
        return
    skip = pytest.mark.skip(
        reason="WHISPER_OFFLINE=1: skip tests that download weights"
    )
    for item in items:
        if "requires_weights" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def default_offline_env(monkeypatch):
    if "WHISPER_OFFLINE" not in os.environ:
        monkeypatch.setenv("WHISPER_OFFLINE", "1")


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


@pytest.fixture
def sample_audio_path():
    path = Path(__file__).resolve().parent / "jfk.flac"
    assert path.is_file()
    assert not str(path).startswith(("http://", "https://"))
    return str(path)


@pytest.fixture(scope="session")
def offline_checkpoint_path(tmp_path_factory):
    """Tiny randomly-initialized checkpoint written at test time. Not in git."""
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=16,
        n_audio_state=16,
        n_audio_head=2,
        n_audio_layer=1,
        n_vocab=32,
        n_text_ctx=16,
        n_text_state=16,
        n_text_head=2,
        n_text_layer=1,
    )
    model = Whisper(dims)
    path = tmp_path_factory.mktemp("offline_weights") / "offline.pt"
    torch.save(
        {"dims": dims.__dict__, "model_state_dict": model.state_dict()},
        path,
    )
    return str(path)
