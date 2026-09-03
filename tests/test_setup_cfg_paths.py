import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_setup_cfg_and_pytest_ini_absent():
    assert not (ROOT / "setup.cfg").exists()
    assert not (ROOT / "setup.py").exists()
    assert not (ROOT / "pytest.ini").exists()


def test_pyproject_names_tests_path():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.pytest.ini_options]" in text
    assert re.search(r"testpaths\s*=\s*\[\s*[\"']tests[\"']\s*\]", text)
    assert re.search(r"exclude\s*=\s*\[\s*[\"']tests\*[\"']\s*\]", text)


def test_documented_test_paths_exist():
    tests = ROOT / "tests"
    for name in (
        "conftest.py",
        "test_audio.py",
        "test_normalizer.py",
        "test_timing.py",
        "test_tokenizer.py",
        "test_transcribe.py",
        "jfk.flac",
    ):
        assert (tests / name).is_file(), name
