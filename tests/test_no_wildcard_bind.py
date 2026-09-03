import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_checker():
    path = REPO_ROOT / "scripts" / "check_no_wildcard_bind.py"
    spec = importlib.util.spec_from_file_location("check_no_wildcard_bind", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_wildcard_checker_flags_production_wildcard(tmp_path):
    check = _load_checker()
    assert check.find_violations(REPO_ROOT, relative_paths=["whisper/serve.py"]) == []
    prod = tmp_path / "whisper"
    prod.mkdir()
    (prod / "bad.py").write_text('app.serve(host="0.0.0.0")\n')
    (prod / "cli.py").write_text("parser.add_argument('--host 0.0.0.0')\n")
    hits = check.find_violations(
        tmp_path, relative_paths=["whisper/bad.py", "whisper/cli.py"]
    )
    patterns = {pattern for _path, pattern in hits}
    assert "0.0.0.0" in patterns
    assert "--host 0.0.0.0" in patterns


def test_wildcard_checker_passes_on_this_repo():
    script = REPO_ROOT / "scripts" / "check_no_wildcard_bind.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK:" in result.stdout
