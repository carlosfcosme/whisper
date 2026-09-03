"""Lock the documented mypy inventory: none in this tree. No weights."""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_pyproject_has_no_mypy():
    text = (ROOT / "pyproject.toml").read_text().lower()
    assert "mypy" not in text
    assert "[tool.mypy" not in text


def test_no_mypy_config_files():
    for name in ("mypy.ini", ".mypy.ini", "setup.cfg"):
        assert not (ROOT / name).exists(), name


def test_precommit_has_no_mypy_hook():
    text = (ROOT / ".pre-commit-config.yaml").read_text().lower()
    assert "mypy" not in text


def test_ci_has_no_mypy_job():
    workflows = ROOT / ".github" / "workflows"
    for path in workflows.glob("*.yml"):
        assert "mypy" not in path.read_text().lower(), path.name
