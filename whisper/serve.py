"""CLI for the loopback-only health server."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .bind import LOOPBACK_HOST, BindError, create_loopback_server


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whisper serve",
        description="Health server bound to 127.0.0.1. Binding 0.0.0.0 is refused.",
    )
    parser.add_argument("--host", default=LOOPBACK_HOST)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    httpd = None
    try:
        httpd = create_loopback_server(args.host, args.port)
        host, port = httpd.server_address[:2]
        print(f"whisper serve listening on http://{host}:{port}", flush=True)
        httpd.serve_forever()
        return 0
    except BindError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0
    finally:
        if httpd is not None:
            httpd.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
