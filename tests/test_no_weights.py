"""Fail if weights are committed or if a pull would be attempted."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import check_no_hub as hub_guard  # noqa: E402
import check_no_weights as guard  # noqa: E402


def test_check_no_weights_passes_on_this_tree():
    assert guard.find_violations(ROOT) == []
    assert guard.main() == 0


def test_classify_rejects_weight_suffixes():
    for suffix in (".pt", ".pth", ".bin", ".ckpt", ".safetensors", ".onnx"):
        assert guard.classify("whisper/tiny{}".format(suffix), 100) is not None


def test_classify_allows_in_repo_fixtures():
    flac = ROOT / "tests" / "jfk.flac"
    npz = ROOT / "whisper" / "assets" / "mel_filters.npz"
    assert guard.classify("tests/jfk.flac", flac.stat().st_size) is None
    assert guard.classify("whisper/assets/mel_filters.npz", npz.stat().st_size) is None


def test_gitignore_covers_weight_caches():
    text = (ROOT / ".gitignore").read_text()
    for pattern in (".cache/", "*.pt", "*.pth", "*.safetensors"):
        assert pattern in text


def test_git_ignores_weight_paths():
    for path in (".cache/whisper/tiny.pt", "tiny.pt", "model.pth"):
        result = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
        assert result.returncode == 0, path


def test_no_hub_ci_script_passes():
    assert hub_guard.main() == 0
