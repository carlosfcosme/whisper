"""CI must fail if model weights or Hub caches are committed."""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

WEIGHT_PATHSPECS = (
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.bin",
    ".cache",
    ".cache/**",
    "weights",
    "weights/**",
    ".huggingface",
    ".huggingface/**",
    "hub",
    "hub/**",
)

IGNORE_EXAMPLES = (
    ".cache/whisper/tiny.pt",
    "weights/tiny.pt",
    ".huggingface/hub/tiny.pt",
    "tiny.pt",
    "model.pth",
    "model.safetensors",
    "model.bin",
)

TRACKED_ASSETS = (
    "whisper/assets/gpt2.tiktoken",
    "whisper/assets/multilingual.tiktoken",
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


def test_no_weight_or_hub_cache_files_tracked():
    listed = _git("ls-files", "-z", "--", *WEIGHT_PATHSPECS)
    assert listed.returncode == 0, listed.stderr
    tracked = [path for path in listed.stdout.split("\0") if path]
    assert tracked == [], "committed weights/caches are not allowed: %s" % tracked


def test_gitignore_keeps_weight_paths_untracked():
    failed = []
    for path in IGNORE_EXAMPLES:
        result = _git("check-ignore", "-q", "--", path)
        if result.returncode != 0:
            failed.append(path)
    assert failed == [], "expected these paths to be gitignored: %s" % failed


def test_tracked_assets_are_not_gitignored():
    for path in TRACKED_ASSETS:
        result = _git("check-ignore", "-q", "--", path)
        assert result.returncode == 1, f"did not expect {path} to be gitignored"
        assert (REPO_ROOT / path).is_file()


def test_package_code_does_not_call_hf_hub():
    listed = _git(
        "grep",
        "-nE",
        "hf_hub_download|snapshot_download|huggingface_hub",
        "--",
        "whisper",
    )
    # git grep returns 1 when there are no matches
    assert listed.returncode == 1, listed.stdout
