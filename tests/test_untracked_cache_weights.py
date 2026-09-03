"""Cache and weight paths must stay untracked.

The library writes named checkpoints to ~/.cache/whisper or
$XDG_CACHE_HOME/whisper (see whisper/__init__.py). --model_dir and
download_root can point inside this checkout. This suite does not
download weights and does not import whisper.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

UNTRACKED_DIR_PREFIXES = (".cache/", "cache/", "weights/")
UNTRACKED_SUFFIXES = (".pt", ".pth")
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
)


def _git(*args: str) -> subprocess.CompletedProcess:
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
    assert missing == [], f"patterns missing from .gitignore: {missing}"


def test_cache_and_weight_paths_are_untracked():
    listed = _git("ls-files", "-z", "--", *LS_FILES_PATHSPECS)
    assert listed.returncode == 0, listed.stderr
    tracked = [path for path in listed.stdout.split("\0") if path]
    assert tracked == [], f"cache/weight paths must stay untracked: {tracked}"


def test_git_index_has_no_cache_or_weight_prefixes():
    listed = _git("ls-files", "-z")
    assert listed.returncode == 0, listed.stderr
    tracked = [path for path in listed.stdout.split("\0") if path]
    offenders = [
        path
        for path in tracked
        if path.startswith(UNTRACKED_DIR_PREFIXES) or path.endswith(UNTRACKED_SUFFIXES)
    ]
    assert offenders == [], f"cache/weight paths must stay untracked: {offenders}"


def test_gitignore_keeps_cache_and_weight_paths_untracked():
    failed = []
    for path in IGNORE_EXAMPLES:
        result = _git("check-ignore", "-q", "--", path)
        if result.returncode != 0:
            failed.append(path)
    assert failed == [], f"expected these paths to be gitignored: {failed}"


def test_tracked_assets_are_not_gitignored():
    for path in TRACKED_ASSETS:
        result = _git("check-ignore", "-q", "--", path)
        assert result.returncode == 1, f"did not expect {path} to be gitignored"
        assert (REPO_ROOT / path).is_file()
