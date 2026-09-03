"""Assert .gitignore covers known Whisper weight/cache path names."""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Repo-relative stand-ins for ~/.cache/whisper and other dump locations.
KNOWN_WEIGHT_CACHE_PATHS = (
    ".cache/whisper/tiny.pt",
    ".cache/whisper/tiny.en.pt",
    ".cache/whisper/base.pt",
    ".cache/huggingface/hub/models--openai--whisper/snapshots/x/model.safetensors",
    ".cache/torch/hub/checkpoints/model.pt",
    "cache/whisper/small.bin",
    "models/tiny.pt",
    "models/large.onnx",
    "weights/base.bin",
    "weights/encoder.onnx",
    "checkpoints/decoder.pt",
    "tiny.pt",
    "model.pth",
    "model.bin",
    "export.onnx",
    "model.safetensors",
)

MUST_STAY_TRACKED = (
    "whisper/__init__.py",
    "whisper/assets/mel_filters.npz",
    "whisper/assets/gpt2.tiktoken",
    "tests/jfk.flac",
    "tests/test_gitignore_weights.py",
    "scripts/check_cache_weights.sh",
    "scripts/check_no_weights.py",
)


def _check_ignore(relpath: str) -> int:
    return subprocess.run(
        ["git", "check-ignore", "-q", "--", relpath],
        cwd=str(REPO_ROOT),
        check=False,
    ).returncode


def test_gitignore_covers_known_weight_cache_paths():
    missed = [path for path in KNOWN_WEIGHT_CACHE_PATHS if _check_ignore(path) != 0]
    assert missed == [], "gitignore does not cover: {}".format(missed)


def test_gitignore_does_not_drop_source_or_fixtures():
    ignored = [path for path in MUST_STAY_TRACKED if _check_ignore(path) == 0]
    assert ignored == [], "gitignore unexpectedly matches: {}".format(ignored)


def test_check_cache_weights_script_passes():
    proc = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "check_cache_weights.sh")],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "OK:" in proc.stdout


def test_git_ls_files_grep_finds_no_weight_blobs():
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    hits = [line for line in proc.stdout.splitlines() if _weight_cache_hit(line)]
    assert hits == [], "tracked weight/cache files: {}".format(hits)


def _weight_cache_hit(relpath: str) -> bool:
    posix = relpath.replace("\\", "/")
    suffixes = (
        ".pt",
        ".pth",
        ".bin",
        ".ckpt",
        ".safetensors",
        ".onnx",
        ".gguf",
        ".ggml",
        ".tflite",
        ".pb",
        ".weights",
    )
    if posix.lower().endswith(suffixes):
        return True
    prefixes = (
        ".cache/",
        "cache/",
        ".huggingface/",
        "huggingface/",
        "weights/",
        "models/",
        "checkpoints/",
    )
    return any(posix.startswith(prefix) for prefix in prefixes)
