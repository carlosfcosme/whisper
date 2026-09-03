"""Inventory of declared pip extras. Does not download model weights."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
EXTRAS_MD = (ROOT / "EXTRAS.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
INSTALL_SH = (ROOT / ".cursor" / "install.sh").read_text(encoding="utf-8")
TEST_YML = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")

DEV_PACKAGES = ("black", "flake8", "isort", "pytest", "scipy")
DEV_LINE = 'optional-dependencies.dev = [ "black", "flake8", "isort", "pytest", "scipy" ]'


def test_only_dev_extra_is_declared():
    extras = re.findall(r"^optional-dependencies\.(\w+)", PYPROJECT, flags=re.M)
    assert extras == ["dev"]
    assert DEV_LINE in PYPROJECT


def test_extras_md_documents_declared_extra():
    assert "one" in EXTRAS_MD.lower()
    assert "`dev`" in EXTRAS_MD
    for package in DEV_PACKAGES:
        assert f"`{package}`" in EXTRAS_MD
    for invented in ("cuda", "gpu", "docs"):
        assert f"| `{invented}`" not in EXTRAS_MD


def test_readme_and_extra_env_name_the_dev_extra():
    assert "[EXTRAS.md](EXTRAS.md)" in README
    assert '".[dev]"' in README
    assert '-e ".[dev]"' in INSTALL_SH
    assert '.["dev"]' in TEST_YML
