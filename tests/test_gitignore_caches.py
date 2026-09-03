"""Weights and cache dirs must be gitignored, not tracked."""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

IGNORE_EXAMPLES = (
    ".cache/whisper/tiny.pt",
    "cache/whisper/tiny.pt",
    "weights/tiny.pt",
    "checkpoints/tiny.pt",
    ".huggingface/hub/tiny.pt",
    "hub/tiny.safetensors",
    "tiny.pt",
    "model.pth",
    "model.safetensors",
    "model.bin",
)

TRACKED_ASSETS = (
    "whisper/assets/gpt2.tiktoken",
    "whisper/assets/multilingual.tiktoken",
    "whisper/assets/mel_filters.npz",
    "tests/jfk.flac",
    "README.md",
)

LS_FILES_PATHSPECS = (
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.bin",
    ".cache",
    ".cache/**",
    "cache",
    "cache/**",
    "weights",
    "weights/**",
    "checkpoints",
    "checkpoints/**",
    ".huggingface",
    ".huggingface/**",
    "hub",
    "hub/**",
)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_weight_and_cache_paths_are_gitignored():
    failed = [
        path
        for path in IGNORE_EXAMPLES
        if _git("check-ignore", "-q", "--", path).returncode != 0
    ]
    assert failed == [], (
        "expected these weight/cache paths to be gitignored: %s" % failed
    )


def test_weight_and_cache_paths_are_untracked():
    listed = _git("ls-files", "-z", "--", *LS_FILES_PATHSPECS)
    assert listed.returncode == 0, listed.stderr
    tracked = [path for path in listed.stdout.split("\0") if path]
    assert tracked == [], "weight/cache paths must stay untracked: %s" % tracked


def test_tracked_assets_are_not_gitignored():
    for path in TRACKED_ASSETS:
        result = _git("check-ignore", "-q", "--", path)
        assert result.returncode == 1, "did not expect %s to be gitignored" % path
        assert (REPO_ROOT / path).is_file()
