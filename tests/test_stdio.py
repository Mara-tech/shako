"""Tests for `core.stdio` — the Windows console encoding repair.

The bug they freeze: `python -m cli` on Windows died in the middle of its progress bar with
`UnicodeEncodeError: 'charmap' codec can't encode characters in position 25-44`, the twenty
positions being the bar's block characters. A cp1252 stream is reproduced here with a plain
`TextIOWrapper`, so the regression is checked on every platform rather than only on the one
that had it.
"""

from __future__ import annotations

import ast
import io
from pathlib import Path
from typing import cast

import pytest

from core.stdio import _CHARACTERS_PRINTED, make_printable, make_stdio_printable

# The bar `cli/main.py` and `rl/self_play.py` draw, at 60 %.
BAR = "█" * 12 + "░" * 8

_ROOT = Path(__file__).resolve().parent.parent

# The modules that put a non-ASCII character on a stream, which `_CHARACTERS_PRINTED` samples.
_PRINTING_MODULES = (
    "cli/main.py",
    "rl/self_play.py",
    "ui/rich_agent.py",
    "ui/grid_widget.py",
)


def _cp1252_stream() -> io.TextIOWrapper:
    """A text stream encoded the way a redirected Windows stdout is."""
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="")


def _written(stream: io.TextIOWrapper) -> str:
    """What actually reached the bytes underneath, read back as UTF-8."""
    stream.flush()
    return cast(io.BytesIO, stream.buffer).getvalue().decode("utf-8")


# -------- the repair itself ----------------------------------------------------


def test_a_cp1252_stream_cannot_take_the_progress_bar() -> None:
    """The report, reproduced — without this the rest of the file proves nothing."""
    stream = _cp1252_stream()
    with pytest.raises(UnicodeEncodeError):
        print(f"\r  Playing games  3/10  {BAR}  30.0%", end="", file=stream, flush=True)


def test_the_repaired_stream_takes_it() -> None:
    stream = _cp1252_stream()
    assert make_printable(stream) is True

    print(f"\r  Playing games  3/10  {BAR}  30.0%", end="", file=stream, flush=True)
    assert BAR in _written(stream)


def test_rich_needs_the_repair_too() -> None:
    """Why the stream is repaired rather than the `print` statement.

    Rich raises the very same error on a cp1252 stream — it even suggests
    PYTHONIOENCODING=utf-8 itself — so rewriting the two bars in ASCII would have left every
    `console.print` of a `★` or a `●` to fail the same way.
    """
    from rich.console import Console

    unrepaired = _cp1252_stream()
    with pytest.raises(UnicodeEncodeError):
        Console(file=unrepaired, force_terminal=False, width=80).print(f"{BAR} ★")

    repaired = _cp1252_stream()
    make_printable(repaired)
    Console(file=repaired, force_terminal=False, width=80).print(f"{BAR} ★")
    assert "★" in _written(repaired)


def test_a_stream_that_can_already_take_them_is_left_alone() -> None:
    """The no-op case, which is every terminal that was not broken to begin with."""
    stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", newline="")
    assert make_printable(stream) is False
    assert stream.encoding == "utf-8"
    assert stream.errors == "strict", "the error handler was relaxed on a healthy stream"


@pytest.mark.parametrize("code_page", ["cp437", "cp850"])
def test_a_dos_code_page_is_repaired_although_it_draws_the_bars(code_page: str) -> None:
    """All or nothing, and this is the case that shows the cost of it.

    cp437 and cp850 carry `█` and `░` natively but not the UI's `★`, so they are switched to
    UTF-8 like any other — losing a rendering they had, rather than keeping a crash they had
    too. A stream has one encoding, not one per character, so there is no third answer.
    """
    assert _encodable("█", code_page) and not _encodable("★", code_page)
    stream = io.TextIOWrapper(io.BytesIO(), encoding=code_page, newline="")
    assert make_printable(stream) is True
    assert stream.encoding == "utf-8"


def _encodable(character: str, encoding: str) -> bool:
    try:
        character.encode(encoding)
    except UnicodeEncodeError:
        return False
    return True


def test_an_unrepairable_stream_is_refused_quietly() -> None:
    """A stream with no `reconfigure` — pytest's own capture is one — must not stop start-up."""

    class Capture:
        encoding = "cp1252"

    assert make_printable(Capture()) is False  # type: ignore[arg-type]
    assert make_printable(object()) is False  # type: ignore[arg-type]


def test_make_stdio_printable_runs_on_the_real_streams() -> None:
    """Called by both entry points, under pytest's capture, it must simply not raise."""
    make_stdio_printable()


# -------- the sample does not rot ----------------------------------------------


def _literals_that_are_not_docstrings(path: Path) -> list[str]:
    """Every string literal of a module except its docstrings — comments are not literals."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    documentation: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            first = body[0].value
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                documentation.add(id(first))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in documentation
    ]


@pytest.mark.parametrize("module", _PRINTING_MODULES)
def test_every_character_these_modules_print_is_in_the_sample(module: str) -> None:
    """A character added to the UI that the sample misses reopens the bug on those consoles.

    Only what cp1252 already refuses matters: `—`, `×` and `·` are in it and travel fine.
    """
    missing = {
        character
        for literal in _literals_that_are_not_docstrings(_ROOT / module)
        for character in literal
        if not _encodable(character, "cp1252") and character not in _CHARACTERS_PRINTED
    }
    assert not missing, f"{module} prints {missing!r}, absent from core.stdio's sample"
