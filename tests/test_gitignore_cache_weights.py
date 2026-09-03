import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATTERNS = (".cache/", "cache/", "weights/", "*.pt", "*.pth")
IGNORE_EXAMPLES = (
    ".cache/whisper/tiny.pt",
    "cache/whisper/tiny.pt",
    "weights/tiny.pt",
    "tiny.pt",
    "model.pth",
)


def test_gitignore_has_cache_and_weight_patterns():
    text = (ROOT / ".gitignore").read_text()
    lines = {line.strip() for line in text.splitlines()}
    missing = [pat for pat in REQUIRED_PATTERNS if pat not in lines]
    assert missing == []


def test_gitignore_ignores_cache_and_weight_examples():
    for path in IGNORE_EXAMPLES:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "--", path],
            cwd=ROOT,
        )
        assert proc.returncode == 0, path


def test_git_does_not_track_cache_or_weights():
    proc = subprocess.run(
        [
            "git",
            "ls-files",
            "--",
            ".cache",
            ".cache/**",
            "cache",
            "cache/**",
            "weights",
            "weights/**",
            "*.pt",
            "*.pth",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == ""
