#!/usr/bin/env python3
"""CI: no-WAN fixture coverage — local fixtures, no model fetch, 127.0.0.1.

Stdlib only. No torch. No Hub. No keys.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOPBACK = "127.0.0.1"
ALL_INTERFACES = ".".join(("0", "0", "0", "0"))


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def _load(relpath: str, name: str):
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        fail(f"could not load {relpath}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_local_fixtures() -> None:
    wan = _load("scripts/check_no_wan_fixtures.py", "check_no_wan_fixtures_ci")
    if wan.main() != 0:
        fail("remote asset URLs in tests")
    jfk = ROOT / "tests" / "jfk.flac"
    if not jfk.is_file():
        fail("missing local cached fixture tests/jfk.flac")
    fixtures = _load("whisper/fixtures.py", "whisper_fixtures_coverage")
    fixtures.require_local_fixture(jfk)
    fixtures.require_local_fixture(fixtures.tiny_fixture_path(generate=False))
    remote = "https://" + "example.com" + "/tiny.pt"
    try:
        fixtures.require_local_fixture(remote)
    except fixtures.RemoteFixtureError:
        pass
    else:
        fail("require_local_fixture must refuse WAN URLs")
    try:
        fixtures.require_local_cached_model(remote)
    except fixtures.RemoteFixtureError:
        pass
    else:
        fail("require_local_cached_model must refuse WAN URLs")


def check_bind_127_0_0_1() -> None:
    bind = _load("whisper/bind.py", "whisper_bind_coverage")
    if bind.require_bind_127_0_0_1(None) != LOOPBACK:
        fail("default bind is not 127.0.0.1")
    try:
        bind.require_bind_127_0_0_1(ALL_INTERFACES)
    except bind.BindError:
        pass
    else:
        fail("bind must refuse all-interfaces")
    start = ROOT / ".cursor" / "start.sh"
    if not start.is_file() or LOOPBACK not in start.read_text(encoding="utf-8"):
        fail(".cursor/start.sh must bind 127.0.0.1")


def check_model_fetch_refused() -> None:
    runtime = _load("whisper/runtime.py", "whisper_runtime_coverage")
    hub = "https://" + "huggingface.co" + "/openai/whisper-tiny/resolve/main/x.bin"
    cdn = "https://" + "openaipublic.azureedge.net" + "/main/whisper/models/ab/tiny.pt"
    for url in (hub, cdn):
        try:
            runtime.refuse_weight_auto_download(url)
        except runtime.WeightDownloadError:
            continue
        fail(f"model fetch was not refused: {url}")
    if runtime.weight_auto_download_allowed():
        fail("weight auto-download must be disabled in CI")


def main() -> int:
    check_local_fixtures()
    check_bind_127_0_0_1()
    check_model_fetch_refused()
    print("no-wan-coverage: ok (local fixtures, no model fetch, 127.0.0.1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
