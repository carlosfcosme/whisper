"""Local-only fixtures for offline CLI tests.

No Hub, no Azure, no committed checkpoints. Audio is the in-repo
``jfk.flac``. Model weights are a tiny runtime checkpoint written to a
temp path (``.pt`` is gitignored and must stay untracked).
"""

from pathlib import Path

import torch

from whisper.model import ModelDimensions, Whisper

JFK_FLAC = Path(__file__).resolve().parent / "jfk.flac"


def toy_checkpoint(path: Path) -> Path:
    """Write a tiny local Whisper checkpoint. Does not download."""
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
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dims": dims.__dict__, "model_state_dict": model.state_dict()}, path)
    return path
