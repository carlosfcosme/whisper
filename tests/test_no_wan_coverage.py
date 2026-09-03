import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(relpath: str, name: str):
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ci_no_wan_coverage_script():
    checker = _load("scripts/check_no_wan_coverage.py", "check_no_wan_coverage")
    assert checker.main() == 0


def test_fail_any_model_fetch_hub_and_cdn():
    runtime = _load("whisper/runtime.py", "runtime_no_wan")
    hub = "https://" + "huggingface.co" + "/models/tiny.pt"
    cdn = "https://" + "openaipublic.azureedge.net" + "/tiny.pt"
    assert runtime.is_remote_model_url(hub)
    assert runtime.is_remote_model_url(cdn)
    with pytest.raises(runtime.WeightDownloadError):
        runtime.refuse_weight_auto_download(hub)
    with pytest.raises(runtime.WeightDownloadError):
        runtime.refuse_weight_auto_download(cdn)
    assert runtime.weight_auto_download_allowed() is False


def test_require_local_cached_fixtures():
    fixtures = _load("whisper/fixtures.py", "fixtures_no_wan")
    audio = fixtures.sample_audio_path()
    tiny = fixtures.tiny_fixture_path(generate=False)
    assert audio.is_file()
    assert tiny.is_file()
    with pytest.raises(fixtures.RemoteFixtureError):
        fixtures.require_local_cached_model("https://" + "example.com" + "/tiny.pt")
    with pytest.raises(FileNotFoundError):
        fixtures.require_local_cached_model("tiny", download_root=ROOT / "tests")


def test_urlopen_wan_model_fetch_is_blocked():
    import urllib.request

    with pytest.raises(RuntimeError, match="WAN"):
        urllib.request.urlopen("https://" + "example.com" + "/tiny.pt", timeout=1)


def test_require_127_0_0_1_bind():
    bind = _load("whisper/bind.py", "bind_no_wan")
    assert bind.require_bind_127_0_0_1(None) == "127.0.0.1"
    all_interfaces = ".".join(("0", "0", "0", "0"))
    with pytest.raises(bind.BindError):
        bind.require_bind_127_0_0_1(all_interfaces)
    with pytest.raises(bind.BindError):
        bind.require_bind_127_0_0_1("")
