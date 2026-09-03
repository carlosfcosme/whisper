#!/usr/bin/env python3
"""CI: validate .cursor/environment.json (ports objects, no Hub, no secrets)."""

import json
import sys
from pathlib import Path

HUB_TOKENS = (
    "huggingface.co",
    "huggingface_hub",
    "hf_hub_download",
    "from_pretrained",
    "hf.co/",
)
SECRET_TOKENS = (
    "api_key",
    "apikey",
    "begin private",
    "password=",
    "secret_key",
)
FORBIDDEN_BIND = ("0.0.0.0", "INADDR_ANY", "inaddr_any")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def validate(root: Path):
    errors = []
    path = root / ".cursor" / "environment.json"
    if not path.is_file():
        return ["missing .cursor/environment.json"]
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ["environment.json is not valid JSON: {}".format(exc)]
    if not isinstance(data, dict):
        return ["environment.json must be an object"]
    if not isinstance(data.get("install"), str) or not data["install"].strip():
        errors.append("install must be a non-empty string")
    if "ports" in data:
        ports = data["ports"]
        if not isinstance(ports, list):
            errors.append("ports must be an array of objects")
        else:
            for i, item in enumerate(ports):
                if not isinstance(item, dict):
                    errors.append(
                        "ports[{}] must be an object {{name, port}}".format(i)
                    )
                    continue
                if "name" not in item or "port" not in item:
                    errors.append("ports[{}] must have name and port".format(i))
                elif not isinstance(item["name"], str) or not isinstance(
                    item["port"], int
                ):
                    errors.append("ports[{}] name must be str and port int".format(i))
                elif not 1 <= item["port"] <= 65535:
                    errors.append("ports[{}] port out of range".format(i))
    lowered = raw.lower()
    for token in HUB_TOKENS:
        if token in lowered:
            errors.append("environment.json must not reference Hub ({})".format(token))
    for token in SECRET_TOKENS:
        if token in lowered:
            errors.append(
                "environment.json must not contain secrets ({})".format(token)
            )
    for token in FORBIDDEN_BIND:
        if token in raw:
            errors.append(
                "environment.json must not contain bind token {}".format(token)
            )
    install_sh = root / ".cursor" / "install.sh"
    if install_sh.is_file():
        script = install_sh.read_text(encoding="utf-8")
        if "load_model" in script:
            errors.append("install.sh must not call load_model")
        if "WHISPER_NO_DOWNLOAD=1" not in script:
            errors.append("install.sh must set WHISPER_NO_DOWNLOAD=1")
    return errors


def main() -> int:
    errors = validate(repo_root())
    if errors:
        sys.stderr.write("ERROR: environment.json / install checks failed:\n")
        for item in errors:
            sys.stderr.write("  {}\n".format(item))
        return 1
    sys.stdout.write("OK: environment.json is valid; install does not fetch models\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
