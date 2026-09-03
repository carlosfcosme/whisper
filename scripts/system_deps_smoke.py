#!/usr/bin/env python3
"""Offline system-dependency smoke: ffmpeg decode + loopback bind.

No model weights. No Hub. No WAN. Stdlib only (does not import torch).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "jfk.flac"
SAMPLE_RATE = 16000
DEFAULT_PORT = 8765


def _load_bind():
    spec = importlib.util.spec_from_file_location(
        "whisper_bind", ROOT / "whisper" / "bind.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_bind = _load_bind()
BIND_HOST = _bind.BIND_HOST
BindError = _bind.BindError
require_bind_127_0_0_1 = _bind.require_bind_127_0_0_1


class SystemDepError(RuntimeError):
    """Raised when a required OS dependency is missing or fails."""


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise SystemDepError(
            "ffmpeg is not on PATH. It is a required OS package and is not "
            "installed by pip."
        )
    return path


def decode_fixture(path=FIXTURE, sr=SAMPLE_RATE) -> bytes:
    """Decode local audio with ffmpeg. Same CLI as whisper.audio.load_audio."""
    if not Path(path).is_file():
        raise SystemDepError("local audio fixture missing: {0}".format(path))
    cmd = [
        require_ffmpeg(),
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise SystemDepError("ffmpeg failed to decode {0}: {1}".format(path, err))
    return proc.stdout


_SMOKE_STATE = {"ffmpeg": False, "pcm_bytes": 0}


def prepare_system_deps() -> bytes:
    """Validate ffmpeg against the local fixture before serving health."""
    pcm = decode_fixture()
    _SMOKE_STATE["ffmpeg"] = True
    _SMOKE_STATE["pcm_bytes"] = len(pcm)
    return pcm


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        payload = json.dumps(
            {
                "ok": bool(_SMOKE_STATE["ffmpeg"]),
                "bind": BIND_HOST,
                "ffmpeg": bool(_SMOKE_STATE["ffmpeg"]),
                "pcm_bytes": _SMOKE_STATE["pcm_bytes"],
                "weights": False,
                "hub": False,
                "offline": True,
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        return


def make_server(host=BIND_HOST, port=0) -> ThreadingHTTPServer:
    require_bind_127_0_0_1(host)
    if not _SMOKE_STATE["ffmpeg"]:
        prepare_system_deps()
    return ThreadingHTTPServer((BIND_HOST, port), _HealthHandler)


def _assert_loopback_url(url: str) -> None:
    host = urlparse(url).hostname
    if host != BIND_HOST:
        raise SystemDepError(
            "smoke may only fetch {0}, got {1!r}".format(BIND_HOST, url)
        )


def run_check() -> int:
    pcm = prepare_system_deps()
    server = make_server(BIND_HOST, 0)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = "http://{0}:{1}/".format(host, port)
    try:
        _assert_loopback_url(url)
        with urllib.request.urlopen(url, timeout=5) as response:
            body = json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()
    if body.get("bind") != BIND_HOST or body.get("weights") or body.get("hub"):
        raise SystemDepError("unexpected smoke health payload: {0}".format(body))
    sys.stdout.write(
        "SMOKE OK ffmpeg={0} bind={1}:{2} pcm_bytes={3}\n".format(
            require_ffmpeg(), host, port, len(pcm)
        )
    )
    return 0


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Offline ffmpeg + 127.0.0.1 smoke (no weights, no Hub)."
    )
    parser.add_argument(
        "--host",
        default=BIND_HOST,
        help="bind address (must be 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="bind port when serving",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="one-shot ffmpeg decode + loopback GET, then exit",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="bind 127.0.0.1 and serve the health endpoint",
    )
    args = parser.parse_args(argv)
    try:
        require_bind_127_0_0_1(args.host)
        if args.check or not args.serve:
            raise SystemExit(run_check())
        prepare_system_deps()
        server = make_server(args.host, args.port)
    except BindError as exc:
        sys.stderr.write("FAIL: {0}\n".format(exc))
        raise SystemExit(2)
    except SystemDepError as exc:
        sys.stderr.write("FAIL: {0}\n".format(exc))
        raise SystemExit(1)
    bound_host, bound_port = server.server_address
    sys.stdout.write("whisper smoke bound to {0}:{1}\n".format(bound_host, bound_port))
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
