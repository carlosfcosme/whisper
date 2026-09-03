from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_license_copyright_year_is_original_publication():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Copyright (c) 2022 OpenAI" in license_text
    # First-publication year; rewriting it to the env calendar year is drift.
    assert "Copyright (c) 2026" not in license_text


def test_readme_python_range_matches_env_classifiers():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "Python 3.8-3.13" in readme
    assert "Python 3.8-3.11" not in readme
    assert 'requires-python = ">=3.8"' in pyproject
    assert "Programming Language :: Python :: 3.13" in pyproject


def test_readme_localhost_only_note():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "localhost-only" in readme
    assert "127.0.0.1" in readme
