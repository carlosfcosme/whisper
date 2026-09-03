"""gitignore must cover weight/cache paths and keep fixtures tracked."""

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_gitignore.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_gitignore", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gitignore_covers_weight_and_cache_paths():
    checker = _load_checker()
    missing, extra = checker.verify(str(ROOT))
    assert missing == []
    assert extra == []
    assert checker.main() == 0


@pytest.mark.parametrize(
    "path",
    [
        "tiny.pt",
        "model.pth",
        ".cache/whisper/tiny.pt",
        "cache/whisper/tiny.pt",
        "model.safetensors",
    ],
)
def test_weight_cache_paths_are_ignored(path):
    checker = _load_checker()
    assert checker.check_ignore(path, str(ROOT)) is True


@pytest.mark.parametrize(
    "path",
    [
        "whisper/__init__.py",
        "whisper/assets/mel_filters.npz",
        "tests/jfk.flac",
        "README.md",
    ],
)
def test_source_and_fixtures_are_not_ignored(path):
    checker = _load_checker()
    assert checker.check_ignore(path, str(ROOT)) is False
