"""Local-only Whisper HTTP server.

Binds 127.0.0.1, defaults to CPU, and loads only local checkpoints.
Hugging Face Hub ids and remote URLs are rejected.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, List, Optional
from urllib.parse import urlparse

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_DEVICE = "cpu"

_HUB_MARKERS = (
    "huggingface.co",
    "hf.co",
    "hf-mirror.com",
    "huggingface_hub",
    "hf://",
    "hf-hub://",
)


@dataclass
class ServeConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    device: str = DEFAULT_DEVICE
    model: Optional[str] = None


def _is_hub_token(part: str) -> bool:
    return bool(part) and part.replace("-", "").replace("_", "").isalnum()


def is_hub_ref(value: str) -> bool:
    lowered = value.strip().lower()
    if any(marker in lowered for marker in _HUB_MARKERS):
        return True
    parsed = urlparse(value)
    if parsed.scheme in {"hf", "hf-hub"}:
        return True
    if os.path.isfile(value) or os.path.isabs(value) or value.startswith("."):
        return False
    parts = value.replace("\\", "/").split("/")
    if len(parts) != 2 or not all(_is_hub_token(part) for part in parts):
        return False
    # Relative checkpoints keep a weight suffix; Hub ids do not.
    weight_suffixes = (".pt", ".pth", ".bin", ".onnx", ".safetensors", ".ckpt")
    return not value.lower().endswith(weight_suffixes)


def is_remote_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https", "ftp"}


def require_loopback_host(host: str) -> str:
    if host != DEFAULT_HOST:
        raise ValueError(f"server must bind to {DEFAULT_HOST}, got {host!r}")
    return host


def require_local_path(path: str, kind: str) -> str:
    if is_hub_ref(path):
        raise ValueError(f"Hugging Face Hub {kind} refs are not supported: {path}")
    if is_remote_url(path):
        raise ValueError(f"remote {kind} URLs are not supported: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"local {kind} not found: {path}")
    return os.path.abspath(path)


def load_local_model(model_path: str, device: str = DEFAULT_DEVICE):
    from . import load_model

    checkpoint = require_local_path(model_path, "checkpoint")
    return load_model(checkpoint, device=device)


class LocalWhisperServer(ThreadingHTTPServer):
    def __init__(self, config: ServeConfig):
        host = require_loopback_host(config.host)
        super().__init__((host, config.port), LocalWhisperHandler)
        self.config = ServeConfig(
            host=host,
            port=config.port,
            device=config.device or DEFAULT_DEVICE,
            model=config.model,
        )
        self.model: Any = None
        if self.config.model:
            self.model = load_local_model(self.config.model, device=self.config.device)


class LocalWhisperHandler(BaseHTTPRequestHandler):
    server_version = "whisper-local/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = posixpath.normpath(urlparse(self.path).path)
        if path == "/health":
            cfg = self.server.config
            self._send_json(
                200,
                {
                    "status": "ok",
                    "host": cfg.host,
                    "port": self.server.server_address[1],
                    "device": cfg.device,
                    "model": cfg.model,
                    "hub": False,
                },
            )
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = posixpath.normpath(urlparse(self.path).path)
        if path not in {"/transcribe", "/v1/audio/transcriptions"}:
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "invalid JSON body"})
            return

        audio = payload.get("audio")
        if not audio or not isinstance(audio, str):
            self._send_json(
                400, {"error": "JSON body must include a local 'audio' path"}
            )
            return

        try:
            audio_path = require_local_path(audio, "audio")
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except FileNotFoundError as exc:
            self._send_json(404, {"error": str(exc)})
            return

        model = self.server.model
        if model is None:
            self._send_json(
                503,
                {"error": "no local checkpoint loaded; pass --model /path/to/model.pt"},
            )
            return

        options = {}
        if payload.get("language"):
            options["language"] = payload["language"]
        result = model.transcribe(audio_path, **options)
        self._send_json(
            200,
            {
                "text": result.get("text", ""),
                "language": result.get("language"),
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whisper-serve",
        description="Serve Whisper on 127.0.0.1 using a local CPU checkpoint.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="bind address (must be 127.0.0.1)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="bind port")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="torch device (defaults to cpu)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="local checkpoint path (Hub ids and URLs are rejected)",
    )
    return parser


def create_server(
    config: Optional[ServeConfig] = None, **kwargs: Any
) -> LocalWhisperServer:
    if config is None:
        config = ServeConfig(**kwargs)
    else:
        for key, value in kwargs.items():
            setattr(config, key, value)
    require_loopback_host(config.host)
    if not config.device:
        config.device = DEFAULT_DEVICE
    return LocalWhisperServer(config)


def cli(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    server = create_server(
        host=args.host,
        port=args.port,
        device=args.device,
        model=args.model,
    )
    bound_host, bound_port = server.server_address
    print(
        "whisper-serve listening on "
        f"http://{bound_host}:{bound_port} (device={args.device})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    cli()
