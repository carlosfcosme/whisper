import os
import subprocess
from pathlib import Path

import whisper

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_download_root_uses_xdg_cache_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert whisper.default_download_root() == os.path.join(str(tmp_path), "whisper")


def test_default_download_root_falls_back_to_home_dot_cache(monkeypatch):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    expected = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
    assert whisper.default_download_root() == expected


def test_readme_documents_cache_dirs_and_gitignore():
    readme = (REPO_ROOT / "README.md").read_text()
    assert "XDG_CACHE_HOME" in readme
    assert "~/.cache/whisper" in readme
    assert "--model_dir" in readme
    assert "download_root" in readme
    assert ".gitignore" in readme


def _is_ignored(relpath: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", relpath],
        cwd=REPO_ROOT,
    )
    return result.returncode == 0


def test_gitignore_covers_cache_dirs_weights_and_secrets():
    ignored = [
        ".cache/whisper/tiny.pt",
        "cache/whisper/base.pt",
        "weights/tiny.pt",
        "tiny.pt",
        "model.pth",
        ".env",
        ".env.local",
    ]
    for path in ignored:
        assert _is_ignored(path), f"expected {path} to be gitignored"

    tracked_assets = [
        "whisper/assets/gpt2.tiktoken",
        "whisper/normalizers/english.json",
        "tests/jfk.flac",
        "README.md",
    ]
    for path in tracked_assets:
        assert not _is_ignored(path), f"did not expect {path} to be gitignored"
        assert (REPO_ROOT / path).is_file()
