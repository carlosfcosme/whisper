#!/usr/bin/env python3
"""Local Whisper demo HTTP server. Always binds loopback."""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, Optional

from whisper.localhost import BIND_HOST, serve_bind_host
from whisper.serve import serve


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Whisper demo server (loopback only)")
    parser.add_argument("--host", default=BIND_HOST)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    serve_bind_host(args.host)
    server = serve(host=args.host, port=args.port)
    host, port = server.server_address[:2]
    sys.stdout.write("Serving on http://%s:%s\n" % (host, port))
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
