"""CI helper must fail when weight files are tracked."""

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_no_weights.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_no_weights", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_repo_has_no_committed_weights():
    checker = _load_checker()
    assert checker.tracked_weight_paths(str(ROOT)) == []
    assert checker.main() == 0


@pytest.mark.parametrize(
    "path,expected",
    [
        ("tiny.pt", True),
        ("weights/model.pth", True),
        ("foo.safetensors", True),
        ("whisper/assets/mel_filters.npz", False),
        ("tests/jfk.flac", False),
        ("whisper/assets/multilingual.tiktoken", False),
    ],
)
def test_is_weight_path(path, expected):
    checker = _load_checker()
    assert checker.is_weight_path(path) is expected


def test_checker_fails_when_weights_are_tracked(monkeypatch, capsys):
    checker = _load_checker()
    monkeypatch.setattr(
        checker, "tracked_weight_paths", lambda repo_root=".": ["tiny.pt"]
    )
    assert checker.main() == 1
    captured = capsys.readouterr()
    assert "tiny.pt" in captured.err
