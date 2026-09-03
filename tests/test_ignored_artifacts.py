import importlib.util
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_checker():
    path = REPO_ROOT / "scripts" / "check_ignored_artifacts.py"
    spec = importlib.util.spec_from_file_location("check_ignored_artifacts", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def test_no_weight_or_cache_paths_are_tracked():
    assert checker.tracked_artifact_paths(REPO_ROOT) == []


def test_gitignore_covers_weight_and_cache_samples():
    assert checker.unignored_samples(REPO_ROOT) == []
    assert checker.main() == 0


def test_git_check_ignore_tiny_pt():
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", "tiny.pt"],
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0


def test_ci_workflow_enforces_ignored_artifacts():
    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    assert "ignored-weight-cache" in workflow
    assert "scripts/check_ignored_artifacts.py" in workflow
