"""Cache and weight paths must stay untracked.

load_model() writes named checkpoints to ~/.cache/whisper or
$XDG_CACHE_HOME/whisper (whisper/__init__.py). --model_dir and
download_root can point inside this checkout. This suite does not
import whisper, download weights, or open a network socket.
Helpers must bind 127.0.0.1 only.
"""

import json
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
TRACKED_ASSETS = (
    "whisper/assets/gpt2.tiktoken",
    "whisper/assets/multilingual.tiktoken",
    "whisper/normalizers/english.json",
    "tests/jfk.flac",
    "README.md",
    ".cursor/environment.json",
    ".cursor/install.sh",
)


def _git(*args):
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_gitignore_declares_cache_and_weight_dirs():
    lines = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    missing = [pattern for pattern in GITIGNORE_PATTERNS if pattern not in lines]
    assert missing == [], "patterns missing from .gitignore: {0}".format(missing)


def test_cache_and_weight_paths_are_untracked():
    listed = _git("ls-files", "-z", "--", *LS_FILES_PATHSPECS)
    assert listed.returncode == 0, listed.stderr
    tracked = [path for path in listed.stdout.split("\0") if path]
    assert tracked == [], "cache/weight paths must stay untracked: {0}".format(tracked)


def test_gitignore_covers_cache_and_weight_examples():
    failed = [
        path
        for path in IGNORE_EXAMPLES
        if _git("check-ignore", "-q", "--", path).returncode != 0
    ]
    assert failed == [], "expected these paths to be gitignored: {0}".format(failed)


def test_tracked_assets_are_not_gitignored():
    for path in TRACKED_ASSETS:
        result = _git("check-ignore", "-q", "--", path)
        assert result.returncode == 1, "did not expect {0} to be gitignored".format(
            path
        )
        assert (REPO_ROOT / path).is_file()


def test_dummy_checkpoints_are_refused_by_git_add():
    for path in IGNORE_EXAMPLES:
        added = _git("add", "-n", "--", path)
        assert added.returncode == 0, added.stderr
        assert path not in added.stdout, added.stdout


def test_environment_is_localhost_only():
    env = json.loads((REPO_ROOT / ".cursor/environment.json").read_text())
    assert "ports" not in env
    assert "start" not in env
    assert "0.0.0.0" not in (REPO_ROOT / ".cursor/install.sh").read_text()
    assert "0.0.0.0" not in (REPO_ROOT / ".github/workflows/test.yml").read_text()
