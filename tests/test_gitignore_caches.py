"""Cache and weight paths must stay untracked."""

import importlib.util
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

GITIGNORE_PATTERNS = (
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.onnx",
    ".cache/",
    ".cache/whisper/",
    ".cache/huggingface/",
    ".huggingface/",
    "hf_cache/",
    "whisper_cache/",
    "checkpoints/",
    "weights/",
    "cache/",
    ".env",
    "*.pem",
    "*.key",
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
    "hf_cache",
    "hf_cache/**",
    "whisper_cache",
    "whisper_cache/**",
    "checkpoints",
    "checkpoints/**",
    "*.pt",
    "*.pth",
    "*.safetensors",
    ".env",
    ".env.*",
)
IGNORE_EXAMPLES = (
    ".cache/whisper/tiny.pt",
    ".cache/huggingface/hub/tiny.pt",
    "cache/whisper/tiny.pt",
    "weights/tiny.pt",
    "hf_cache/tiny.safetensors",
    "whisper_cache/tiny.pt",
    "checkpoints/tiny.pt",
    "tiny.pt",
    "model.pth",
    "model.safetensors",
    ".env",
    ".env.local",
    "secrets.pem",
    "id_rsa.key",
)
TRACKED_ASSETS = (
    "whisper/assets/gpt2.tiktoken",
    "whisper/assets/multilingual.tiktoken",
    "whisper/assets/mel_filters.npz",
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
    assert missing == [], "patterns missing from .gitignore: {}".format(missing)


def test_check_gitignore_script_agrees():
    path = REPO_ROOT / "scripts" / "check_gitignore_caches.py"
    spec = importlib.util.spec_from_file_location("check_gitignore_caches", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.missing_patterns(REPO_ROOT) == []
    assert module.unignored_examples(REPO_ROOT) == []
    assert module.wrongly_ignored_assets(REPO_ROOT) == []


def test_cache_and_weight_paths_are_untracked():
    listed = _git("ls-files", "-z", "--", *LS_FILES_PATHSPECS)
    assert listed.returncode == 0, listed.stderr
    tracked = [path for path in listed.stdout.split("\0") if path]
    assert tracked == [], "cache/weight paths must stay untracked: {}".format(tracked)


def test_gitignore_keeps_cache_and_weight_paths_untracked():
    failed = []
    for path in IGNORE_EXAMPLES:
        if _git("check-ignore", "-q", "--", path).returncode != 0:
            failed.append(path)
    assert failed == [], "expected these paths to be gitignored: {}".format(failed)


def test_tracked_assets_are_not_gitignored():
    for path in TRACKED_ASSETS:
        result = _git("check-ignore", "-q", "--", path)
        assert result.returncode == 1, "did not expect {} to be gitignored".format(path)
        assert (REPO_ROOT / path).is_file()
