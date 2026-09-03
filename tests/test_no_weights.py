import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_checker():
    path = REPO / "scripts" / "check_no_weights.py"
    spec = importlib.util.spec_from_file_location("check_no_weights", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def test_current_tree_has_no_committed_weights():
    assert checker.find_violations(REPO) == []
    assert checker.main() == 0


def test_planted_checkpoint_is_a_violation(tmp_path):
    planted = tmp_path / "tiny.pt"
    planted.write_bytes(b"not-a-real-checkpoint")
    violations = checker.find_violations(tmp_path, ["tiny.pt"])
    assert violations == [("tiny.pt", "model weight or checkpoint (.pt)")]


def test_classify_weight_suffixes():
    assert checker.classify("models/tiny.pth", 10) is not None
    assert checker.classify("models/model.safetensors", 10) is not None
    assert checker.classify("models/weights.bin", 10) is not None
    assert checker.classify("whisper/transcribe.py", 10) is None
    assert checker.classify("whisper/assets/mel_filters.npz", 4271) is None


def test_check_script_exits_zero_on_clean_repo():
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "check_no_weights.py")],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "OK:" in result.stdout


def test_repo_root_matches_checkout():
    assert checker.repo_root() == REPO


def test_gitignore_covers_cache_and_weights():
    text = (REPO / ".gitignore").read_text()
    for pattern in (
        "*.pt",
        "*.pth",
        "*.safetensors",
        ".cache/",
        "**/.cache/whisper/",
        ".huggingface/",
    ):
        assert pattern in text


def test_assert_no_weight_download_clean(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    spec_path = REPO / "scripts" / "assert_no_weight_download.py"
    spec = importlib.util.spec_from_file_location(
        "assert_no_weight_download", spec_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.find_downloads() == []
    assert module.main() == 0


def test_assert_no_weight_download_detects_cache(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    xdg = tmp_path / "xdg"
    (xdg / "whisper").mkdir(parents=True)
    (xdg / "whisper" / "tiny.pt").write_bytes(b"checkpoint")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    spec_path = REPO / "scripts" / "assert_no_weight_download.py"
    spec = importlib.util.spec_from_file_location(
        "assert_no_weight_download_dirty", spec_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    hits = module.find_downloads()
    assert any(path.endswith("tiny.pt") for path in hits)
    assert module.main() == 1
