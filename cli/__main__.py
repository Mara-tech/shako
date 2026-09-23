from cli.main import main
from core.stdio import make_stdio_printable


if __name__ == "__main__":
    # Before anything prints: the progress bar and the UI draw characters cp1252 has no room
    # for, and a redirected stdout on Windows is encoded that way. See core/stdio.py.
    make_stdio_printable()
    main()
