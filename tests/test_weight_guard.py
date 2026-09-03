import importlib.util
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_guard():
    path = REPO_ROOT / "scripts" / "check_weight_guard.py"
    spec = importlib.util.spec_from_file_location("check_weight_guard", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_guard_passes_on_this_repo():
    guard = _load_guard()
    assert guard.run_checks(REPO_ROOT) == []
    proc = subprocess.run(
        ["python3", str(REPO_ROOT / "scripts" / "check_weight_guard.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_tracked_weight_path_fails(tmp_path):
    guard = _load_guard()
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "ci@example.com")
    _git(tmp_path, "config", "user.name", "CI")
    weight = tmp_path / "tiny.pt"
    weight.write_bytes(b"not-a-checkpoint")
    _git(tmp_path, "add", "tiny.pt")
    assert "tiny.pt" in guard.tracked_weight_hits(tmp_path)
    errors = guard.run_checks(tmp_path)
    assert any("tracked" in err for err in errors)


def test_tracked_cache_path_fails(tmp_path):
    guard = _load_guard()
    cache = tmp_path / ".cache" / "whisper"
    cache.mkdir(parents=True)
    (cache / "tiny.pt").write_bytes(b"not-a-checkpoint")
    assert guard.is_weight_or_cache_path(".cache/whisper/tiny.pt")


def test_install_curl_weights_fails(tmp_path):
    guard = _load_guard()
    script = tmp_path / "install.sh"
    script.write_text(
        "#!/bin/sh\n"
        "curl -fsSL https://openaipublic.azureedge.net/main/whisper/models/x/tiny.pt "
        "-o tiny.pt\n"
    )
    hits = guard.install_weight_fetch_hits([script])
    assert hits == [str(script)]


def test_commented_weight_fetch_is_ignored(tmp_path):
    guard = _load_guard()
    script = tmp_path / "install.sh"
    script.write_text("# curl https://example.com/tiny.pt -o tiny.pt\necho ok\n")
    assert guard.install_weight_fetch_hits([script]) == []


def test_current_install_does_not_fetch_weights():
    guard = _load_guard()
    install = REPO_ROOT / ".cursor" / "install.sh"
    assert install.is_file()
    assert guard.install_weight_fetch_hits([install]) == []
    uncommented = "\n".join(guard._uncommented_lines(install.read_text(encoding="utf-8")))
    assert "load_model" not in uncommented
    assert "azureedge" not in uncommented


def test_start_script_localhost_only():
    guard = _load_guard()
    start = REPO_ROOT / ".cursor" / "start.sh"
    assert start.is_file()
    assert "127.0.0.1" in start.read_text(encoding="utf-8")
    assert guard.all_interface_hits([start]) == []
