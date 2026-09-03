import sys

if len(sys.argv) >= 2 and sys.argv[1] == "serve":
    from .serve import main as serve_main

    raise SystemExit(serve_main(sys.argv[2:]))

from .transcribe import cli

cli()
