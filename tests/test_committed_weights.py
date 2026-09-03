"""Committed checkpoints must stay out of git. Does not download weights."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = REPO_ROOT / ".github" / "scripts" / "check_no_committed_weights.py"

GITIGNORE_PATTERNS = (
    ".cache/",
    "cache/",
    "weights/",
    "*.pt",
    "*.pth",
    "*.safetensors",
)
IGNORE_EXAMPLES = (
    ".cache/whisper/tiny.pt",
    "cache/whisper/tiny.pt",
    "weights/tiny.pt",
    "tiny.pt",
    "model.pth",
    "model.safetensors",
)


def _git(*args):
    return subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
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
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT)],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "OK: no committed model weights" in result.stdout


def test_weight_paths_are_gitignored():
    failed = [
        path
        for path in IGNORE_EXAMPLES
        if _git("check-ignore", "-q", "--", path).returncode != 0
    ]
    assert failed == [], failed
