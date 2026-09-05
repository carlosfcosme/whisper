#!/usr/bin/env python3
"""Fail CI if model weights or large binaries are committed.

Weights are downloaded at runtime (or loaded from a local cache). They must
not be tracked in git.
"""

import subprocess
import sys
from pathlib import Path

WEIGHT_SUFFIXES = frozenset(
    {
        ".pt",
        ".pth",
        ".bin",
        ".ckpt",
        ".safetensors",
        ".onnx",
        ".h5",
        ".hdf5",
        ".gguf",
        ".ggml",
        ".tflite",
        ".pb",
        ".mar",
        ".params",
        ".weights",
    }
)

BINARY_SUFFIXES = frozenset(
    {".so", ".dylib", ".dll", ".exe", ".whl", ".egg", ".a", ".o"}
)

# Official Whisper checkpoints start around 75 MiB. Tracked fixtures stay
# under this limit (largest is notebooks/Multilingual_ASR.ipynb, ~5.7 MiB).
MAX_FILE_BYTES = 10 * 1024 * 1024


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def tracked_files(root: Path):
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=str(root))
    return [
        path for path in output.decode("utf-8", "surrogateescape").split("\0") if path
    ]


def classify(relpath: str, size: int):
    suffix = Path(relpath.replace("\\", "/")).suffix.lower()
    if suffix in WEIGHT_SUFFIXES:
        return "model weight or checkpoint ({})".format(suffix)
    if suffix in BINARY_SUFFIXES:
        return "committed binary ({})".format(suffix)
    if size > MAX_FILE_BYTES:
        return "large file ({} bytes > {})".format(size, MAX_FILE_BYTES)
    return None


def find_violations(root: Path, relative_paths=None):
    paths = relative_paths if relative_paths is not None else tracked_files(root)
    violations = []
    for relpath in paths:
        path = root / relpath
        if not path.is_file():
            continue
        reason = classify(relpath, path.stat().st_size)
        if reason:
            violations.append((relpath, reason))
    return violations


def main() -> int:
    root = repo_root()
    violations = find_violations(root)
    if violations:
        sys.stderr.write(
            "ERROR: model weights or large binaries must not be committed:\n"
        )
        for relpath, reason in violations:
            sys.stderr.write("  {}: {}\n".format(relpath, reason))
        return 1
    sys.stdout.write("OK: no model weights or oversized binaries in the git tree\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
