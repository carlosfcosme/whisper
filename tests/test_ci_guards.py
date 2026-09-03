import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bind_checker_passes_on_this_repo():
    checker = _load_script("check_bind_localhost.py")
    assert checker.main() == 0


def test_bind_checker_fails_when_all_interfaces_injected(tmp_path):
    checker = _load_script("check_bind_localhost.py")
    whisper_dir = tmp_path / "whisper"
    whisper_dir.mkdir()
    all_interfaces = ".".join(("0", "0", "0", "0"))
    (whisper_dir / "evil.py").write_text(f'HOST = "{all_interfaces}"\n')
    hits = checker.find_all_interface_hits(tmp_path)
    assert hits
    assert any("evil.py" in hit for hit in hits)


def test_no_hub_checker_passes_on_this_repo():
    checker = _load_script("check_no_hub.py")
    assert checker.main() == 0


def test_no_weights_checker_passes_on_this_repo():
    checker = _load_script("check_no_weights.py")
    assert checker.main() == 0


def test_no_wan_fixture_checker_passes_on_this_repo():
    checker = _load_script("check_no_wan_fixtures.py")
    assert checker.main() == 0


def test_bind_module_refuses_all_interfaces_without_package_import():
    checker = _load_script("check_bind_localhost.py")
    bind = checker.load_bind_module()
    all_interfaces = ".".join(("0", "0", "0", "0"))
    with pytest.raises(bind.BindError):
        bind.require_bind_127_0_0_1(all_interfaces)
    assert bind.require_bind_127_0_0_1(None) == "127.0.0.1"
