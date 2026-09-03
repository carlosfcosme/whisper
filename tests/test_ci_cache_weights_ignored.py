"""CI: verify cache and weight artifacts are gitignored and untracked."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WEIGHT_SAMPLES = (
    "tiny.pt",
    "models/large.pth",
    "checkpoint.ckpt",
    "model.safetensors",
    "weights/model.gguf",
    ".cache/whisper/tiny.pt",
    "whisper/.cache/model.pt",
)


def _load_checker(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ignored(relpath: str) -> bool:
    proc = subprocess.run(
        ["git", "check-ignore", "-q", "--", relpath],
        cwd=ROOT,
        check=False,
    )
    return proc.returncode == 0


def test_weight_and_cache_paths_are_gitignored():
    missing = [path for path in WEIGHT_SAMPLES if not _ignored(path)]
    assert missing == [], f"not gitignored: {missing}"


def test_git_ls_files_has_no_weight_suffixes():
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    tracked = [p for p in output.decode("utf-8", "surrogateescape").split("\0") if p]
    forbidden = {".pt", ".pth", ".ckpt", ".safetensors", ".gguf", ".onnx", ".ggml"}
    leaked = [
        path
        for path in tracked
        if Path(path).suffix.lower() in forbidden
        and path != "whisper/assets/mel_filters.npz"
    ]
    assert leaked == []


def test_gitignore_declares_weight_and_cache_patterns():
    text = (ROOT / ".gitignore").read_text()
    for pattern in ("*.pt", "*.safetensors", ".cache/whisper/"):
        assert pattern in text


def test_check_gitignore_weights_script_passes():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_gitignore_weights.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_check_no_weights_script_passes():
    checker = _load_checker("check_no_weights.py")
    assert checker.findings(ROOT) == []
    assert checker.main() == 0


def test_check_no_weights_flags_a_planted_checkpoint(tmp_path):
    checker = _load_checker("check_no_weights.py")
    (tmp_path / "leaked.pt").write_bytes(b"x")
    issues = checker.findings(tmp_path, paths=["leaked.pt"])
    assert issues
    assert "leaked.pt" in issues[0]
