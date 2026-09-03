"""Cache and weight paths must stay untracked."""

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_gitignore_weights.py"

GITIGNORE_PATTERNS = (
    ".cache/",
    "cache/",
    "weights/",
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.ckpt",
)
IGNORE_EXAMPLES = (
    ".cache/whisper/tiny.pt",
    "cache/whisper/tiny.pt",
    "weights/tiny.pt",
    "tiny.pt",
    "model.pth",
    "model.safetensors",
)
TRACKED_ASSETS = (
    "whisper/assets/gpt2.tiktoken",
    "whisper/assets/multilingual.tiktoken",
    "whisper/normalizers/english.json",
    "tests/jfk.flac",
    "README.md",
)


def _load_check():
    spec = importlib.util.spec_from_file_location("check_gitignore_weights", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_checker_subprocess_passes_on_this_tree():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "gitignored" in proc.stdout


def test_gitignore_declares_cache_and_weight_patterns():
    lines = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    missing = [pattern for pattern in GITIGNORE_PATTERNS if pattern not in lines]
    assert missing == [], "patterns missing from .gitignore: %s" % missing


def test_cache_and_weight_paths_are_untracked():
    check = _load_check()
    assert check.tracked_weight_paths() == []


def test_gitignore_keeps_cache_and_weight_paths_untracked():
    failed = []
    for path in IGNORE_EXAMPLES:
        if _git("check-ignore", "-q", "--", path).returncode != 0:
            failed.append(path)
    assert failed == [], "expected these paths to be gitignored: %s" % failed


def test_tracked_assets_are_not_gitignored():
    for path in TRACKED_ASSETS:
        result = _git("check-ignore", "-q", "--", path)
        assert result.returncode == 1, "did not expect %s to be gitignored" % path
        assert (REPO_ROOT / path).is_file()


def test_checker_detects_missing_pattern():
    check = _load_check()
    missing = check.missing_gitignore_patterns(".cache/\ncache/\n")
    assert "*.pt" in missing
    assert "weights/" in missing


def test_workflow_runs_gitignore_weights_job():
    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    assert "gitignore-weights:" in workflow
    assert "scripts/check_gitignore_weights.py" in workflow
    assert "HF_HUB_OFFLINE" in workflow
    assert "not test_transcribe" in workflow
