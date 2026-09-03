"""Cache and weight paths must stay untracked.

The library writes named checkpoints to ~/.cache/whisper or
$XDG_CACHE_HOME/whisper (see whisper/__init__.py). --model_dir and
download_root can point inside this checkout. This suite does not
download weights and does not contact the Hugging Face Hub.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_cache_weights.sh"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"
PRECOMMIT = REPO_ROOT / ".pre-commit-config.yaml"

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
GOOD_GITIGNORE = ".cache/\ncache/\nweights/\n*.pt\n*.pth\n"


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def _run_script(cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def _init_repo(tmp_path: Path, gitignore_text: str) -> None:
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    (tmp_path / ".gitignore").write_text(gitignore_text)
    subprocess.run(
        ["git", "add", "--", ".gitignore"],
        cwd=tmp_path,
        check=True,
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
    listed = _git(REPO_ROOT, "ls-files", "-z", "--", *LS_FILES_PATHSPECS)
    assert listed.returncode == 0, listed.stderr
    tracked = [path for path in listed.stdout.split("\0") if path]
    assert tracked == [], f"cache/weight paths must stay untracked: {tracked}"


def test_git_index_has_no_cache_or_weight_prefixes():
    listed = _git(REPO_ROOT, "ls-files", "-z")
    assert listed.returncode == 0, listed.stderr
    tracked = [path for path in listed.stdout.split("\0") if path]
    offenders = [
        path
        for path in tracked
        if path.startswith((".cache/", "cache/", "weights/"))
        or path.endswith((".pt", ".pth"))
    ]
    assert offenders == [], f"cache/weight paths must stay untracked: {offenders}"


def test_gitignore_keeps_cache_and_weight_paths_untracked():
    failed = []
    for path in IGNORE_EXAMPLES:
        result = _git(REPO_ROOT, "check-ignore", "-q", "--", path)
        if result.returncode != 0:
            failed.append(path)
    assert failed == [], f"expected these paths to be gitignored: {failed}"


def test_tracked_assets_are_not_gitignored():
    for path in TRACKED_ASSETS:
        result = _git(REPO_ROOT, "check-ignore", "-q", "--", path)
        assert result.returncode == 1, f"did not expect {path} to be gitignored"
        assert (REPO_ROOT / path).is_file()


def test_git_add_refuses_dummy_checkpoint():
    copied = REPO_ROOT / "_ci_guard_dummy.pt"
    copied.write_bytes(b"not-a-real-weight")
    try:
        dry = _git(REPO_ROOT, "add", "-n", "--", copied.name)
        assert dry.returncode == 0, dry.stderr
        assert copied.name not in dry.stdout
        listed = _git(REPO_ROOT, "ls-files", "--", copied.name)
        assert listed.stdout.strip() == ""
    finally:
        copied.unlink(missing_ok=True)
        _git(REPO_ROOT, "reset", "-q", "HEAD", "--", copied.name)


def test_check_script_is_present():
    assert SCRIPT.is_file()
    text = SCRIPT.read_text()
    assert "git ls-files" in text
    assert "Hugging Face Hub" in text
    assert "*.pt" in text


def test_ci_and_precommit_invoke_the_check_script():
    workflow = WORKFLOW.read_text()
    assert "cache-weights:" in workflow
    assert "scripts/check_cache_weights.sh" in workflow
    assert "HF_HUB_OFFLINE" in workflow
    assert PRECOMMIT.read_text().count("scripts/check_cache_weights.sh") == 1


def test_check_script_passes_on_this_repo():
    result = _run_script(REPO_ROOT)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_check_script_fails_when_pt_is_tracked(tmp_path):
    _init_repo(tmp_path, GOOD_GITIGNORE)
    (tmp_path / "tiny.pt").write_bytes(b"not-a-real-weight")
    subprocess.run(
        ["git", "add", "-f", "--", "tiny.pt"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    result = _run_script(tmp_path)
    assert result.returncode != 0
    assert "tiny.pt" in result.stderr


def test_check_script_fails_when_cache_dir_file_is_tracked(tmp_path):
    _init_repo(tmp_path, GOOD_GITIGNORE)
    cache_dir = tmp_path / ".cache" / "whisper"
    cache_dir.mkdir(parents=True)
    (cache_dir / "tiny.pt").write_bytes(b"not-a-real-weight")
    subprocess.run(
        ["git", "add", "-f", "--", ".cache/whisper/tiny.pt"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    result = _run_script(tmp_path)
    assert result.returncode != 0
    assert ".cache/whisper/tiny.pt" in result.stderr


def test_check_script_fails_when_weights_dir_file_is_tracked(tmp_path):
    _init_repo(tmp_path, GOOD_GITIGNORE)
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "model.pth").write_bytes(b"not-a-real-weight")
    subprocess.run(
        ["git", "add", "-f", "--", "weights/model.pth"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    result = _run_script(tmp_path)
    assert result.returncode != 0
    assert "weights/model.pth" in result.stderr


def test_check_script_fails_when_gitignore_omits_patterns(tmp_path):
    _init_repo(tmp_path, "# no cache or weight rules\n")
    result = _run_script(tmp_path)
    assert result.returncode != 0
    assert "missing required pattern" in result.stderr
