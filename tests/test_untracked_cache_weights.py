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
    created = []
    try:
        for rel in IGNORE_EXAMPLES:
            path = REPO_ROOT / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not-a-real-checkpoint")
            created.append(path)
        status = _git("status", "--porcelain", "--", *IGNORE_EXAMPLES)
        assert status.returncode == 0, status.stderr
        assert status.stdout.strip() == "", status.stdout
        added = _git("add", "-n", "--", *IGNORE_EXAMPLES)
        assert added.returncode != 0, added.stdout
        assert added.stdout.strip() == "", added.stdout
        assert "ignored" in added.stderr
    finally:
        for path in created:
            path.unlink()
            parent = path.parent
            while parent != REPO_ROOT and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent


def test_environment_is_localhost_only():
    env = json.loads((REPO_ROOT / ".cursor/environment.json").read_text())
    assert "ports" not in env
    assert "start" not in env
    install = (REPO_ROOT / ".cursor/install.sh").read_text()
    assert "0.0.0.0" not in install
    bind_tokens = ("--host", " bind", "listen")
    bind_hits = [
        line
        for line in (REPO_ROOT / ".github/workflows/test.yml").read_text().splitlines()
        if "0.0.0.0" in line and any(token in line for token in bind_tokens)
    ]
    assert bind_hits == [], "CI must not bind 0.0.0.0: {0}".format(bind_hits)
