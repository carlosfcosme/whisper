"""Ignore-coverage and git ls-files guard for cache/weight blobs."""

import runpy
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_tracked_weights.py"

MUST_IGNORE = (
    "tiny.pt",
    "tiny.en.pt",
    "model.pth",
    "pytorch_model.bin",
    "model.safetensors",
    "model.onnx",
    "model.ckpt",
    "model.gguf",
    "model.ggml",
    ".cache/whisper/tiny.pt",
    ".cache/huggingface/hub/models--openai--whisper-tiny/blobs/abc",
    "cache/whisper/base.pt",
    "weights/large-v3.pt",
    "checkpoints/epoch0.ckpt",
    ".huggingface/hub/models--x/snapshots/y/model.safetensors",
    "huggingface/hub/models--openai--whisper/snapshots/z/model.safetensors",
)

MUST_KEEP = (
    "whisper/model.py",
    "whisper/__init__.py",
    "whisper/assets/mel_filters.npz",
    "whisper/assets/gpt2.tiktoken",
    "whisper/assets/multilingual.tiktoken",
    "tests/jfk.flac",
    "tests/test_audio.py",
    "pyproject.toml",
    ".github/workflows/test.yml",
)


def _mod():
    return runpy.run_path(str(SCRIPT))


def _is_ignored(path):
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", path],
        cwd=str(ROOT),
    )
    return result.returncode == 0


def test_gitignore_lists_required_patterns():
    text = (ROOT / ".gitignore").read_text()
    for pattern in _mod()["GITIGNORE_PATTERNS"]:
        assert pattern in text, pattern


def test_gitignore_covers_weight_and_cache_paths():
    missing = [path for path in MUST_IGNORE if not _is_ignored(path)]
    assert missing == [], missing


def test_gitignore_keeps_source_and_fixtures():
    blocked = [path for path in MUST_KEEP if _is_ignored(path)]
    assert blocked == [], blocked


def test_git_ls_files_guard_clean_on_this_repo():
    ns = _mod()
    assert ns["git_ls_files"](ns["PATHSPECS"]) == []
    assert ns["main"]() == 0


def test_git_ls_files_guard_fails_when_pt_tracked(tmp_path):
    ns = _mod()
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True)
    subprocess.run(
        ["git", "config", "user.email", "ci@example.test"],
        cwd=str(tmp_path),
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "ci"],
        cwd=str(tmp_path),
        check=True,
    )
    (tmp_path / "ok.py").write_text("x\n")
    (tmp_path / "evil.pt").write_bytes(b"not-a-real-checkpoint")
    subprocess.run(["git", "add", "ok.py", "evil.pt"], cwd=str(tmp_path), check=True)
    tracked = ns["git_ls_files"](ns["PATHSPECS"], cwd=str(tmp_path))
    assert "evil.pt" in tracked
    assert "ok.py" not in tracked
