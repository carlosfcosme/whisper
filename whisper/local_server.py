"""Local Whisper HTTP server. Binds 127.0.0.1 only. CPU default. No Hub."""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlparse

from .defaults import (
    DEFAULT_DEVICE,
    DEFAULT_HOST,
    DEFAULT_PORT,
    reject_huggingface_hub,
    require_loopback_host,
)


class LocalWhisperHandler(BaseHTTPRequestHandler):
    server_version = "WhisperLocal/1.0"

    def log_message(self, format, *args):
        return

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/health"):
            self._send_json(
                200,
                {
                    "status": "ok",
                    "host": self.server.server_address[0],
                    "device": self.server.device,
                    "hub": False,
                },
            )
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/transcribe":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "invalid json"})
            return

        model = payload.get("model", "")
        audio = payload.get("audio", "")
        if not isinstance(model, str) or not isinstance(audio, str):
            self._send_json(400, {"error": "model and audio must be strings"})
            return

        try:
            reject_huggingface_hub(model)
            reject_huggingface_hub(audio)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        from . import load_model, resolve_local_checkpoint, transcribe

        try:
            checkpoint = resolve_local_checkpoint(model)
        except FileNotFoundError as exc:
            self._send_json(404, {"error": str(exc)})
            return
        except (ValueError, RuntimeError) as exc:
            self._send_json(400, {"error": str(exc)})
            return

        if not audio or not os.path.isfile(audio):
            self._send_json(400, {"error": "local audio file is required"})
            return

        model_obj = load_model(checkpoint, device=self.server.device, local_only=True)
        result = transcribe(model_obj, audio)
        self._send_json(200, {"text": result.get("text", "")})


class LocalWhisperServer(ThreadingHTTPServer):
    def __init__(self, host: str, port: int, device: str = DEFAULT_DEVICE):
        host = require_loopback_host(host)
        super().__init__((host, port), LocalWhisperHandler)
        self.device = device or DEFAULT_DEVICE


def create_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    device: str = DEFAULT_DEVICE,
) -> LocalWhisperServer:
    return LocalWhisperServer(host, port, device=device)


def cli(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Local Whisper server (127.0.0.1)")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    args = parser.parse_args(argv)
    server = create_server(args.host, args.port, device=args.device)
    bind_host, bind_port = server.server_address
    print(
        f"whisper local server http://{bind_host}:{bind_port} device={server.device}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    cli()
