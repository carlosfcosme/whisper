import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "serve":
        from .serve import main as serve_main

        serve_main(argv[1:])
        return
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    from .transcribe import cli

    cli()


if __name__ == "__main__":
    main()
