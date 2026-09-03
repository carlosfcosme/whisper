import importlib.util
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_no_weights.py"
GITIGNORE_PATTERNS = (".cache/", "cache/", "weights/", "*.pt", "*.pth")
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
    "whisper/assets/mel_filters.npz",
    "whisper/normalizers/english.json",
    "tests/jfk.flac",
    "README.md",
)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )


def _load_check():
    spec = importlib.util.spec_from_file_location("check_no_weights", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gitignore_declares_cache_and_weight_patterns():
    lines = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    missing = [pattern for pattern in GITIGNORE_PATTERNS if pattern not in lines]
    assert missing == [], "patterns missing from .gitignore: %s" % missing


def test_gitignore_ignores_weight_examples():
    failed = []
    for path in IGNORE_EXAMPLES:
        result = _git("check-ignore", "-q", "--", path)
        if result.returncode != 0:
            failed.append(path)
    assert failed == [], "expected these paths to be gitignored: %s" % failed


def test_tracked_assets_are_not_gitignored():
    for path in TRACKED_ASSETS:
        result = _git("check-ignore", "-q", "--", path)
        assert result.returncode == 1, "did not expect %s to be gitignored" % path
        assert (REPO_ROOT / path).is_file()


def test_git_index_has_no_weight_or_cache_artifacts():
    listed = _git("ls-files", "-z")
    assert listed.returncode == 0, listed.stderr
    tracked = [path for path in listed.stdout.split("\0") if path]
    check = _load_check()
    assert check.classify_tracked(tracked) == []


def test_checker_reports_missing_gitignore(tmp_path, monkeypatch):
    check = _load_check()
    (tmp_path / ".gitignore").write_text("*.pyc\n")
    monkeypatch.setattr(check, "ROOT", tmp_path)
    missing = check.missing_gitignore_patterns(tmp_path)
    assert "*.pt" in missing
    assert check.main() == 1


def test_checker_reports_tracked_checkpoint():
    check = _load_check()
    hits = check.classify_tracked(
        ["whisper/tiny.pt", "README.md", ".cache/whisper/base.pt"]
    )
    assert hits == ["whisper/tiny.pt", ".cache/whisper/base.pt"]


def test_checker_subprocess_passes_on_this_tree():
    proc = subprocess.run(
        ["python3", str(SCRIPT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_ci_skips_transcribe_and_runs_weight_guard():
    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    assert "scripts/check_no_weights.py" in workflow
    assert "scripts/check_bind_localhost.py" in workflow
    assert "-k 'not test_transcribe'" in workflow
    assert "test_transcribe[tiny]" not in workflow
    assert "WHISPER_OFFLINE" in workflow


def test_install_does_not_download_weights():
    text = (REPO_ROOT / ".cursor" / "install.sh").read_text(encoding="utf-8")
    assert "load_model" not in text
    assert "precache" not in text
