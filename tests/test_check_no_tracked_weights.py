"""The check script must fail CI when cache or weight files are tracked."""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_no_tracked_weights.sh"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"

GOOD_GITIGNORE = ".cache/\ncache/\nweights/\n*.pt\n*.pth\n"


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


def test_check_script_is_present_and_executable_via_bash():
    assert SCRIPT.is_file()
    text = SCRIPT.read_text()
    assert "git ls-files" in text
    assert "*.pt" in text


def test_ci_and_precommit_invoke_the_check_script():
    workflow = WORKFLOW.read_text()
    assert "scripts/check_no_tracked_weights.sh" in workflow
    precommit = (REPO_ROOT / ".pre-commit-config.yaml").read_text()
    assert "scripts/check_no_tracked_weights.sh" in precommit


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
