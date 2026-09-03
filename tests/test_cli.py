"""Lock the whisper CLI entrypoint. Does not download model weights."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
MAIN = (ROOT / "whisper" / "__main__.py").read_text(encoding="utf-8")
TRANSCRIBE = (ROOT / "whisper" / "transcribe.py").read_text(encoding="utf-8")
CLI_MD = (ROOT / "CLI.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
INSTALL_SH = (ROOT / ".cursor" / "install.sh").read_text(encoding="utf-8")
ENTRYPOINT = 'scripts.whisper = "whisper.transcribe:cli"'


def test_pyproject_registers_whisper_cli():
    assert ENTRYPOINT in PYPROJECT


def test_egginfo_console_script_matches():
    entry_points = ROOT / "openai_whisper.egg-info" / "entry_points.txt"
    if not entry_points.is_file():
        return
    text = entry_points.read_text(encoding="utf-8")
    assert "[console_scripts]" in text
    assert "whisper = whisper.transcribe:cli" in text


def test_cli_function_is_importable():
    from whisper.transcribe import cli

    assert callable(cli)
    assert cli.__name__ == "cli"


def test_main_module_calls_cli():
    assert "from .transcribe import cli" in MAIN
    assert "cli()" in MAIN
    assert 'if __name__ == "__main__":' in TRANSCRIBE
    assert "def cli():" in TRANSCRIBE
    assert TRANSCRIBE.find("parse_args") < TRANSCRIBE.find("load_model")


def test_readme_and_install_point_at_cli_md():
    assert "[CLI.md](CLI.md)" in README
    assert "whisper.transcribe:cli" in README
    assert "python3 -m whisper" in README
    assert "CLI.md" in INSTALL_SH
    assert "whisper.transcribe:cli" in INSTALL_SH


def test_cli_md_documents_working_invocations():
    assert (
        ENTRYPOINT in CLI_MD or 'scripts.whisper = "whisper.transcribe:cli"' in CLI_MD
    )
    assert "`whisper …`" in CLI_MD or "`whisper`" in CLI_MD
    assert "python3 -m whisper" in CLI_MD
    assert "whisper/__main__.py" in CLI_MD
    assert "--help" in CLI_MD
    assert "load_model" in CLI_MD
    assert "attempted relative import" in CLI_MD
    assert "Do not download or commit checkpoints" in CLI_MD


def test_help_exits_zero_without_weights(tmp_path):
    cache = tmp_path / "xdg-cache"
    cache.mkdir()
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(cache)
    commands = [
        [sys.executable, "-m", "whisper", "--help"],
    ]
    whisper_bin = shutil.which("whisper")
    if whisper_bin:
        commands.insert(0, [whisper_bin, "--help"])

    for cmd in commands:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
        assert "--model" in result.stdout
        assert "turbo" in result.stdout

    assert list(cache.rglob("*")) == []
    assert list(cache.rglob("*.pt")) == []


def test_transcribe_py_as_script_fails_relative_import():
    result = subprocess.run(
        [sys.executable, str(ROOT / "whisper" / "transcribe.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "attempted relative import" in result.stderr
