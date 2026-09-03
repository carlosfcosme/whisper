"""Cache/weight gitignore tests. No torch, no model download."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CACHE_DIR_NAMES = (
    ".cache/",
    ".cache/whisper/",
    "cache/",
    "cache/whisper/",
    "weights/",
)
WEIGHT_NAMES = (
    ".cache/whisper/tiny.pt",
    "cache/whisper/base.pt",
    "weights/model.pth",
    "tiny.pt",
    "model.pth",
    "model.safetensors",
    "model.ckpt",
    "model.onnx",
)
KEEP_TRACKED = (
    "tests/jfk.flac",
    "whisper/assets/multilingual.tiktoken",
    "whisper/assets/gpt2.tiktoken",
    "whisper/__init__.py",
    ".gitignore",
)


def test_gitignore_lists_cache_and_weight_patterns():
    text = (REPO / ".gitignore").read_text(encoding="utf-8")
    for pattern in (
        ".cache/",
        "cache/",
        "weights/",
        "*.pt",
        "*.pth",
        "*.safetensors",
        "*.ckpt",
        "*.onnx",
    ):
        assert pattern in text, pattern


def test_git_ignores_cache_directories_and_weight_files():
    samples = CACHE_DIR_NAMES + WEIGHT_NAMES
    out = subprocess.check_output(
        ["git", "check-ignore", "-v", *samples], cwd=REPO, text=True
    )
    for sample in samples:
        assert sample in out, sample


def test_source_fixtures_are_not_gitignored():
    proc = subprocess.run(
        ["git", "check-ignore", "-v", *KEEP_TRACKED],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    for path in KEEP_TRACKED:
        assert (REPO / path).is_file()
        assert path not in proc.stdout


def test_git_add_does_not_stage_weight_files():
    weight = REPO / "ci-gitignore-probe.pt"
    try:
        weight.write_bytes(b"not-a-real-checkpoint")
        subprocess.run(
            ["git", "add", "ci-gitignore-probe.pt"],
            cwd=REPO,
            check=False,
            capture_output=True,
            text=True,
        )
        tracked = subprocess.check_output(
            ["git", "ls-files", "ci-gitignore-probe.pt"], cwd=REPO, text=True
        ).strip()
        assert tracked == ""
    finally:
        if weight.exists():
            weight.unlink()
        subprocess.run(
            ["git", "reset", "-q", "--", "ci-gitignore-probe.pt"],
            cwd=REPO,
            check=False,
            capture_output=True,
        )
