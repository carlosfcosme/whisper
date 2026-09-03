import sys

if len(sys.argv) > 1 and sys.argv[1] == "serve":
    from .serve import main

    raise SystemExit(main(sys.argv[2:]))

from .transcribe import cli

cli()
