"""Keep the process's standard streams able to carry what this project prints.

The CLI and the trainer draw progress bars out of block characters, and the Rich/Textual
UI marks seats and tiles with `★`, `●`, `▼`. None of those exist in cp1252, so writing one
to a stream encoded that way raises `UnicodeEncodeError` mid-print:

    UnicodeEncodeError: 'charmap' codec can't encode characters in position 25-44

which is what `python -m cli` did on Windows, in `_run_simulation`'s progress bar.

**Rich does not shield anything here** — measured on a cp1252 stream, `console.print` raises
the same error and adds its own advice, "You may need to add PYTHONIOENCODING=utf-8 to your
environment". So the fix cannot live in one `print` statement, and swapping the bars for
ASCII would still leave the UI's marks to fail. The streams are what is wrong, so the streams
are what is repaired, once, at each entry point.

**Why a Windows console reaches cp1252 at all**: since Python 3.6 a real console goes through
`_WindowsConsoleIO` in UTF-8, so it does not. Falling back to the locale encoding means stdout
is *not* a console — it is redirected to a file or a pipe, or it is an IDE's run window. Those
read UTF-8 perfectly well, which is what makes UTF-8 the right thing to reconfigure to.

**It is a no-op wherever nothing is broken**: the stream is touched only when its encoding
refuses one of the characters below, so a UTF-8 terminal is left exactly as it was. It is all
or nothing, a stream having one encoding and not one per character — so the DOS code pages
cp437 and cp850, which carry `█` and `░` but not `★`, are switched as well, and the price they
pay is the native rendering of the bars they could already draw.
"""

from __future__ import annotations

import sys
from typing import TextIO

# Every non-ASCII character this project writes to a stream, and where from:
#   █ ░  the progress bars of `cli/main.py` and `rl/self_play.py`
#   ★    the seat mark of `ui/rich_agent.py`
#   ● ▼  the tokens and the hovered column of `ui/grid_widget.py`
# A character added to that list without being added here costs nothing on a UTF-8 stream,
# and reopens this bug on a stream that has every character above but not the new one.
_CHARACTERS_PRINTED = "█░★●▼"


def make_printable(stream: TextIO) -> bool:
    """Switch `stream` to UTF-8 if its encoding cannot carry `_CHARACTERS_PRINTED`.

    Returns True when the stream was changed, False when it was already fine or could not be
    changed. Never raises: a stream this cannot repair is a stream the program should still
    start on — it fails later, at the first character it cannot write, which is exactly where
    it failed before.

    `errors="replace"` rather than the default, so that a console that ends up unable to show
    a character prints a substitute instead of killing the run at the last line of a report.
    """
    encoding = getattr(stream, "encoding", None)
    reconfigure = getattr(stream, "reconfigure", None)
    if not isinstance(encoding, str) or reconfigure is None:
        return False  # not a text stream we can ask, e.g. a capture object under pytest
    try:
        _CHARACTERS_PRINTED.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        pass
    else:
        return False  # it already carries everything; leave it exactly as it is
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return False
    return True


def make_stdio_printable() -> None:
    """Apply `make_printable` to stdout and stderr. Call it from an entry point, once."""
    for stream in (sys.stdout, sys.stderr):
        make_printable(stream)
