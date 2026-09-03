import importlib.util
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK = os.path.join(ROOT, "scripts", "check_no_weights.py")


def _load_check():
    spec = importlib.util.spec_from_file_location("check_no_weights", CHECK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_no_weights = _load_check()


def test_classify_rejects_weight_suffixes():
    assert check_no_weights.classify("tiny.pt", 100) is not None
    assert check_no_weights.classify("model.pth", 100) is not None
    assert check_no_weights.classify("weights/tiny.safetensors", 100) is not None
    assert check_no_weights.classify("cache/whisper/tiny.pt", 10) is not None
    assert check_no_weights.classify(".cache/whisper/tiny.pt", 10) is not None


def test_classify_allows_in_repo_fixtures():
    jfk = os.path.join("tests", "jfk.flac")
    size = os.path.getsize(os.path.join(ROOT, jfk))
    assert check_no_weights.classify(jfk, size) is None
    assert check_no_weights.classify("whisper/assets/mel_filters.npz", 5000) is None
    assert check_no_weights.classify("whisper/assets/gpt2.tiktoken", 800000) is None


def test_find_violations_flags_weight_paths(tmp_path):
    (tmp_path / "tiny.pt").write_bytes(b"not-a-real-checkpoint")
    hits = check_no_weights.find_violations(tmp_path, relative_paths=["tiny.pt"])
    assert hits and hits[0][0] == "tiny.pt"


def test_repo_has_no_tracked_weights():
    assert check_no_weights.find_violations(os.path.abspath(ROOT)) == []


def test_gitignore_covers_weight_globs():
    text = open(os.path.join(ROOT, ".gitignore"), encoding="utf-8").read()
    for pattern in check_no_weights.REQUIRED_GITIGNORE:
        assert pattern in text, pattern


def test_git_ignores_weight_and_cache_paths():
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "-v",
            "tiny.pt",
            "model.pth",
            "cache/whisper/tiny.pt",
            ".cache/whisper/tiny.pt",
            "weights/tiny.safetensors",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    ignored = {line.split()[-1] for line in result.stdout.splitlines() if line}
    assert "tiny.pt" in ignored
    assert "cache/whisper/tiny.pt" in ignored


def test_check_no_weights_script_passes():
    result = subprocess.run(
        [sys.executable, CHECK],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
