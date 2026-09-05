"""Verify weight/cache paths are gitignored and untracked."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SAMPLES = (
    "tiny.pt",
    "models/large.pth",
    "model.safetensors",
    ".cache/whisper/tiny.pt",
    "whisper/.cache/model.pt",
)


def test_weight_paths_are_gitignored():
    missing = []
    for relpath in SAMPLES:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "--", relpath],
            cwd=ROOT,
            check=False,
        )
        if proc.returncode != 0:
            missing.append(relpath)
    assert missing == [], f"not gitignored: {missing}"


def test_no_tracked_weight_suffixes():
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    tracked = [p for p in output.decode("utf-8", "surrogateescape").split("\0") if p]
    forbidden = {".pt", ".pth", ".ckpt", ".safetensors", ".gguf", ".onnx", ".ggml"}
    leaked = [p for p in tracked if Path(p).suffix.lower() in forbidden]
    assert leaked == []
