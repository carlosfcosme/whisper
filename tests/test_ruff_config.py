"""Ruff config is present and the check command is wired. No weights."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ruff_toml_is_present():
    path = REPO_ROOT / "ruff.toml"
    assert path.is_file()
    text = path.read_text()
    assert "line-length = 88" in text
    assert 'target-version = "py38"' in text


def test_ci_runs_ruff_check():
    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text()
    assert "ruff check" in workflow
    assert "ruff.toml" in workflow
