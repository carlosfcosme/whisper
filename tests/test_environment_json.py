import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_environment_json_exists_and_parses():
    path = ROOT / ".cursor" / "environment.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data.get("name"), str) and data["name"]
    assert data.get("install") == "bash .cursor/install.sh"


def test_environment_json_ports_omitted_or_objects():
    data = json.loads((ROOT / ".cursor" / "environment.json").read_text())
    if "ports" not in data:
        return
    assert isinstance(data["ports"], list)
    for item in data["ports"]:
        assert isinstance(item, dict)
        assert isinstance(item["name"], str)
        assert isinstance(item["port"], int)
        assert 1 <= item["port"] <= 65535


def test_environment_json_has_no_hub_or_wildcard_bind():
    raw = (ROOT / ".cursor" / "environment.json").read_text()
    lowered = raw.lower()
    tokens = (
        "huggingface" + ".co",
        "huggingface" + "_hub",
        "from_" + "pretrained",
        "hf." + "co/",
        ".".join(("0",) * 4),
    )
    for token in tokens:
        assert token not in lowered


def test_environment_ci_script_passes():
    path = ROOT / "scripts" / "check_environment_json.py"
    spec = importlib.util.spec_from_file_location("check_environment_json", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.validate(ROOT) == []
    assert module.main() == 0


def test_install_sh_does_not_download_models():
    script = (ROOT / ".cursor" / "install.sh").read_text()
    assert "load_model" not in script
    assert "WHISPER_NO_DOWNLOAD=1" in script
    assert "HF_HUB_OFFLINE=1" in script
