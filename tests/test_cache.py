import os

import whisper


def test_documented_cache_dir_names():
    assert whisper._CACHE_DIR_NAME == "whisper"
    assert whisper._CACHE_HOME_ENV == "XDG_CACHE_HOME"
    assert whisper._DEFAULT_CACHE_HOME_NAME == ".cache"


def test_default_download_root_uses_xdg_cache_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert whisper.default_download_root() == os.path.join(str(tmp_path), "whisper")


def test_default_download_root_falls_back_to_home_dot_cache(monkeypatch):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    expected = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
    assert whisper.default_download_root() == expected
