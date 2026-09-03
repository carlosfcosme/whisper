"""CI weight guard and gitignore cache coverage. No torch, no Hub."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import check_no_weights as guard  # noqa: E402


def test_check_no_weights_passes_on_this_tree():
    assert guard.find_violations(ROOT) == []
    assert guard.main() == 0


def test_classify_rejects_weight_suffixes():
    for suffix in (".pt", ".pth", ".bin", ".ckpt", ".safetensors", ".onnx", ".gguf"):
        assert guard.classify("whisper/tiny{}".format(suffix), 100) is not None


def test_classify_allows_in_repo_fixtures():
    flac = ROOT / "tests" / "jfk.flac"
    npz = ROOT / "whisper" / "assets" / "mel_filters.npz"
    assert guard.classify("tests/jfk.flac", flac.stat().st_size) is None
    assert guard.classify("whisper/assets/mel_filters.npz", npz.stat().st_size) is None


def test_classify_rejects_oversized_file():
    assert guard.classify("blob.dat", guard.MAX_FILE_BYTES + 1) is not None


def test_gitignore_covers_caches_and_weights():
    text = (ROOT / ".gitignore").read_text()
    for pattern in (
        "__pycache__/",
        ".pytest_cache",
        ".cache/",
        "cache/",
        "*.pt",
        "*.pth",
        "*.safetensors",
    ):
        assert pattern in text


def test_git_ignores_cache_and_weight_paths():
    paths = [
        ".cache/whisper/tiny.pt",
        "cache/whisper/tiny.pt",
        "tiny.pt",
        "model.pth",
        ".pytest_cache/v/cache",
        "__pycache__/x.pyc",
    ]
    for path in paths:
        result = subprocess.run(
            ["git", "check-ignore", "-q", path],
            cwd=ROOT,
        )
        assert result.returncode == 0, path
