"""Local Whisper HTTP server: CPU default, 127.0.0.1 only, no Hub."""

import argparse
import json
import os
import tempfile
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .sources import (
    DEFAULT_BIND_HOST,
    default_device,
    reject_hub_source,
    require_loopback_host,
)

DEFAULT_PORT = 8000
DEFAULT_MODEL = "tiny"


class LoopbackHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class ServerState:
    def __init__(
        self,
        model_name=DEFAULT_MODEL,
        device=None,
        download_root=None,
        model=None,
        loader=None,
    ):
        reject_hub_source(model_name)
        reject_hub_source(download_root)
        self.model_name = model_name
        self.device = default_device() if device is None else device
        self.download_root = download_root
        self._model = model
        self._loader = loader

    @property
    def model(self):
        if self._model is None:
            if self._loader is not None:
                self._model = self._loader()
            else:
                from . import load_model

                self._model = load_model(
                    self.model_name,
                    device=self.device,
                    download_root=self.download_root,
                )
        return self._model

    def health(self, host, port):
        return {
            "status": "ok",
            "device": str(self.device),
            "host": host,
            "port": port,
            "model": self.model_name,
            "hub": False,
        }


class WhisperRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, state=None, **kwargs):
        self.state = state
        BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

    def _json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/health"):
            host, port = self.server.server_address[:2]
            self._json(200, self.state.health(host, port))
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/transcribe":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._json(400, {"error": "empty body"})
            return
        body = self.rfile.read(length)
        audio = _extract_audio(self.headers.get("Content-Type") or "", body)
        if not audio:
            self._json(400, {"error": "no audio in request"})
            return
        tmp = tempfile.NamedTemporaryFile(
            prefix="whisper-server-", suffix=".audio", delete=False
        )
        try:
            tmp.write(audio)
            tmp.close()
            result = self.state.model.transcribe(tmp.name)
        except Exception as exc:
            self._json(500, {"error": str(exc)})
            return
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
        self._json(
            200,
            {
                "text": result.get("text", ""),
                "language": result.get("language"),
                "segments": result.get("segments", []),
            },
        )


def _extract_audio(content_type, body):
    if content_type.startswith("multipart/"):
        from email import policy
        from email.parser import BytesParser

        preamble = ("Content-Type: {}\r\n\r\n".format(content_type)).encode("utf-8")
        message = BytesParser(policy=policy.default).parsebytes(preamble + body)
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if (
                name in ("file", "audio")
                or part.get_content_disposition() == "attachment"
            ):
                payload = part.get_payload(decode=True)
                if payload:
                    return payload
        return None
    return body or None


def create_server(
    host=DEFAULT_BIND_HOST, port=DEFAULT_PORT, state=None, **state_kwargs
):
    host = require_loopback_host(host)
    if state is None:
        state = ServerState(**state_kwargs)
    handler = partial(WhisperRequestHandler, state=state)
    return LoopbackHTTPServer((host, port), handler)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a local Whisper HTTP server (CPU, 127.0.0.1, no Hub)."
    )
    parser.add_argument("--host", default=DEFAULT_BIND_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=default_device())
    parser.add_argument(
        "--model_dir",
        default=None,
        help="local directory for official checkpoints; Hub paths are rejected",
    )
    return parser


def cli(argv=None):
    args = build_parser().parse_args(argv)
    httpd = create_server(
        host=args.host,
        port=args.port,
        model_name=args.model,
        device=args.device,
        download_root=args.model_dir,
    )
    host, port = httpd.server_address[:2]
    print(
        "whisper-server listening on http://{}:{} (device={}, hub=off)".format(
            host, port, args.device
        )
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    cli()
