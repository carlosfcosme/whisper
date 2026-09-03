"""Cache and weight paths must stay untracked."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

UNTRACKED_DIR_PREFIXES = (".cache/", "cache/", "weights/", ".huggingface/", "hub/")
UNTRACKED_SUFFIXES = (".pt", ".pth", ".safetensors", ".bin")
GITIGNORE_PATTERNS = (
    ".cache/",
    "cache/",
    "weights/",
    ".huggingface/",
    "hub/",
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.bin",
    ".env",
    ".env.*",
)
LS_FILES_PATHSPECS = (
    ".cache",
    ".cache/**",
    "cache",
    "cache/**",
    "weights",
    "weights/**",
    ".huggingface",
    ".huggingface/**",
    "hub",
    "hub/**",
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.bin",
    ".env",
    ".env.*",
)
IGNORE_EXAMPLES = (
    ".cache/whisper/tiny.pt",
    "cache/whisper/tiny.pt",
    "weights/tiny.pt",
    ".huggingface/hub/tiny.pt",
    "hub/tiny.safetensors",
    "tiny.pt",
    "model.pth",
    "model.safetensors",
    "model.bin",
    ".env",
    ".env.local",
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


def test_ci_verify_gitignore_script():
    script = REPO_ROOT / ".github" / "scripts" / "verify-gitignore.py"
    assert script.is_file()
    listed = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert listed.returncode == 0, listed.stderr
    assert "ok: gitignore" in listed.stdout
