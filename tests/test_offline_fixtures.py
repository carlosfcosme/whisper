import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_fixtures():
    path = ROOT / "whisper" / "fixtures.py"
    spec = importlib.util.spec_from_file_location("whisper_fixtures_offline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tiny_fixture_is_local_and_tiny():
    fixtures = _load_fixtures()
    path = fixtures.tiny_fixture_path(generate=False)
    assert path.is_file()
    assert path.suffix == ".wav"
    assert path.stat().st_size < 8 * 1024
    resolved = fixtures.require_local_fixture(path)
    assert resolved == path.resolve()


def test_jfk_fixture_is_local_not_url():
    fixtures = _load_fixtures()
    path = fixtures.sample_audio_path()
    assert path.is_file()
    assert path.name == "jfk.flac"
    assert not fixtures.is_remote_asset_url(str(path))


@pytest.mark.parametrize(
    "url",
    [
        "https://" + "example.com" + "/audio.flac",
        "http://" + "openaipublic.azureedge.net" + "/tiny.pt",
        "https://"
        + "huggingface.co"
        + "/openai/whisper-tiny/resolve/main/pytorch_model.bin",
    ],
)
def test_require_local_fixture_refuses_wan_urls(url):
    fixtures = _load_fixtures()
    with pytest.raises(fixtures.RemoteFixtureError):
        fixtures.require_local_fixture(url)


def test_wan_fixture_checker_passes_on_this_repo():
    checker = _load_script("check_no_wan_fixtures.py")
    assert checker.main() == 0


def test_wan_fixture_checker_fails_on_remote_url(tmp_path):
    checker = _load_script("check_no_wan_fixtures.py")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    remote = "https://" + "example.com" + "/sample.flac"
    (tests_dir / "bad.py").write_text(
        'AUDIO = "{}"\n'.format(remote),
        encoding="utf-8",
    )
    hits = checker.find_remote_asset_urls(tmp_path)
    assert hits
    assert any("example.com" in url for _, url in hits)
