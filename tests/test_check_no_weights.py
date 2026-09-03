import importlib.util
import os
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_no_weights.py"


def _checker():
    spec = importlib.util.spec_from_file_location("check_no_weights", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    return _checker()


@pytest.mark.parametrize(
    "path",
    [
        "tiny.pt",
        "models/base.pth",
        "pytorch_model.bin",
        "model.safetensors",
        "weights/foo.ckpt",
        "export.onnx",
        "ggml-tiny.bin",
        ".cache/whisper/tiny.pt",
        "huggingface/hub/models--openai--whisper/snapshots/x/model.safetensors",
    ],
)
def test_weight_paths_are_rejected(checker, path):
    assert checker.is_committed_weight(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "whisper/assets/mel_filters.npz",
        "whisper/assets/multilingual.tiktoken",
        "whisper/assets/gpt2.tiktoken",
        "tests/jfk.flac",
        "whisper/__init__.py",
        "README.md",
        "approach.png",
    ],
)
def test_source_and_assets_are_allowed(checker, path):
    assert checker.is_committed_weight(path) is False


def test_repo_has_no_committed_weights(checker):
    bad = checker.find_committed_weights()
    assert bad == []


def test_checker_main_fails_on_extra_weight(checker, capsys):
    rc = checker.main(["oops.pt"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "oops.pt" in err


def test_checker_script_is_executable_without_whisper():
    assert _SCRIPT.is_file()
    assert os.access(str(_SCRIPT), os.R_OK)
