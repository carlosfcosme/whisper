import runpy
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "check_no_committed_weights.py"
)


def _mod():
    return runpy.run_path(str(SCRIPT))


def test_detects_weight_suffixes():
    is_weight = _mod()["is_committed_weight"]
    assert is_weight("models/tiny.pt")
    assert is_weight("cache/pytorch_model.bin")
    assert is_weight("dir/model.safetensors")
    assert is_weight("weights/foo.onnx")
    assert not is_weight("whisper/model.py")
    assert not is_weight("whisper/assets/mel_filters.npz")
    assert not is_weight("whisper/assets/gpt2.tiktoken")


def test_iter_tracked_weights_filters():
    iter_weights = _mod()["iter_tracked_weights"]
    found = list(
        iter_weights(
            [
                "whisper/__init__.py",
                "models/tiny.pt",
                "whisper/assets/gpt2.tiktoken",
            ]
        )
    )
    assert found == ["models/tiny.pt"]


def test_repo_has_no_committed_weights():
    ns = _mod()
    tracked = list(ns["iter_tracked_weights"](ns["git_tracked_paths"]()))
    assert tracked == []
