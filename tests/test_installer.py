"""Lock the documented installer: pip, not uv. Does not download model weights."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALLER_MD = (ROOT / "INSTALLER.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
REQUIREMENTS = (ROOT / "requirements.txt").read_text(encoding="utf-8")
INSTALL_SH = (ROOT / ".cursor" / "install.sh").read_text(encoding="utf-8")
TEST_YML = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
PUBLISH_YML = (ROOT / ".github" / "workflows" / "python-publish.yml").read_text(
    encoding="utf-8"
)

UV_LOCK_NAMES = ("uv.lock", "Pipfile", "poetry.lock")
SECRET_PATTERNS = (
    "API_KEY",
    "SECRET_KEY",
    "PRIVATE_KEY",
    "BEGIN RSA",
    "sk-",
)


def test_repo_has_no_uv_or_alt_lockfiles():
    for name in UV_LOCK_NAMES:
        assert not (ROOT / name).exists()
    assert "[tool.uv]" not in PYPROJECT
    assert not re.search(r"\buv\b", PYPROJECT)
    assert not re.search(r"\buv\b", REQUIREMENTS)


def test_bootstrap_and_ci_call_pip_not_uv():
    assert "pip install --break-system-packages" in INSTALL_SH
    assert '-e ".[dev]"' in INSTALL_SH
    assert not re.search(r"\buv\b", INSTALL_SH)
    assert "pip install --upgrade pre-commit" in TEST_YML
    assert "pip3 install" in TEST_YML
    assert not re.search(r"\buv\b", TEST_YML)
    assert "python -m pip install --upgrade pip" in PUBLISH_YML
    assert not re.search(r"\buv\b", PUBLISH_YML)


def test_docs_name_pip_and_reject_uv():
    assert "[INSTALLER.md](INSTALLER.md)" in README
    assert "pip install -U openai-whisper" in README
    assert "**pip**" in INSTALLER_MD
    assert "**uv**" in INSTALLER_MD
    assert "uv.lock" in INSTALLER_MD
    assert "[tool.uv]" in INSTALLER_MD
    assert "uv pip install" in INSTALLER_MD
    assert "uv sync" in INSTALLER_MD


def test_installer_docs_have_no_weights_or_secrets():
    weight_files = [path for path in ROOT.rglob("*.pt") if ".git" not in path.parts]
    assert weight_files == []
    for pattern in SECRET_PATTERNS:
        assert pattern not in INSTALLER_MD
    assert "load_model" not in INSTALLER_MD
