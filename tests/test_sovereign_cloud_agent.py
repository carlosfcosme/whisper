"""Sovereign Cloud Agent path: no WAN, no weight pull, localhost bind.

These checks read `.cursor/` files only. They do not import whisper,
download checkpoints, or open a listening socket.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL = (REPO_ROOT / ".cursor/install.sh").read_text()
VERIFY = (REPO_ROOT / ".cursor/verify.sh").read_text()
ENV = json.loads((REPO_ROOT / ".cursor/environment.json").read_text())

WEIGHT_FETCH = (
    "whisper.load_model",
    "load_model(",
    "openaipublic.azureedge.net",
    "urllib.request.urlopen",
)


def _code_lines(text):
    return [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_environment_json_is_localhost_only():
    assert ENV.get("install") == "bash .cursor/install.sh"
    assert "ports" not in ENV
    assert "start" not in ENV
    assert "0.0.0.0" not in json.dumps(ENV)


def test_install_sh_never_pulls_weights():
    code = "\n".join(_code_lines(INSTALL))
    for token in WEIGHT_FETCH:
        assert token not in code, "install.sh must not {0}".format(token)
    assert "0.0.0.0" not in INSTALL


def test_install_sh_skips_wan_when_deps_ready():
    assert "deps_ready" in INSTALL
    assert "sovereign: deps present" in INSTALL
    assert re.search(r"^if deps_ready; then", INSTALL, re.M)
    apt_lines = [line for line in _code_lines(INSTALL) if "apt-get" in line]
    pip_lines = [line for line in _code_lines(INSTALL) if "pip install" in line]
    assert apt_lines, "bootstrap apt-get should remain for first-time setup"
    assert pip_lines, "bootstrap pip should remain for first-time setup"
    # WAN package installs live only in the else/bootstrap branch.
    before_else, after_else = INSTALL.split("else", 1)
    assert "apt-get" not in before_else
    assert "pip install" not in before_else
    assert "apt-get" in after_else
    assert "pip install" in after_else


def test_verify_sh_is_sovereign():
    code = "\n".join(_code_lines(VERIFY))
    for token in WEIGHT_FETCH:
        assert token not in code, "verify.sh must not {0}".format(token)
    assert "not test_transcribe" in VERIFY
    assert "127.0.0.1" in VERIFY
    assert "XDG_CACHE_HOME" in VERIFY
    assert "http://127.0.0.1:9" in VERIFY
    assert "0.0.0.0" not in VERIFY


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("SOVEREIGN STATIC OK")
