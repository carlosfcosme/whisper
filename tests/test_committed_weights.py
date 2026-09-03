"""Committed checkpoints must stay out of git. Does not download weights."""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

GITIGNORE_PATTERNS = (".cache/", "cache/", "weights/", "*.pt", "*.pth")
LS_FILES_PATHSPECS = (
    ".cache",
    ".cache/**",
    "cache",
    "cache/**",
    "weights",
    "weights/**",
    "*.pt",
    "*.pth",
)
IGNORE_EXAMPLES = (
    ".cache/whisper/tiny.pt",
    "cache/whisper/tiny.pt",
    "weights/tiny.pt",
    "tiny.pt",
    "model.pth",
)


def _git(*args):
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_gitignore_declares_weight_paths():
    lines = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    missing = [pattern for pattern in GITIGNORE_PATTERNS if pattern not in lines]
    assert missing == [], missing


def test_git_has_no_committed_weights():
    listed = _git("ls-files", "-z", "--", *LS_FILES_PATHSPECS)
    assert listed.returncode == 0, listed.stderr
    tracked = [path for path in listed.stdout.split("\0") if path]
    assert tracked == [], "committed weights: {0}".format(tracked)


def test_weight_paths_are_gitignored():
    failed = [
        path
        for path in IGNORE_EXAMPLES
        if _git("check-ignore", "-q", "--", path).returncode != 0
    ]
    assert failed == [], failed


def test_cache_and_weights_dirs_are_ignored():
    for directory in ("cache/", "weights/", "cache/whisper/tiny.pt", "weights/tiny.pt"):
        result = _git("check-ignore", "-q", "--", directory)
        assert result.returncode == 0, directory
