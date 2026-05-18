from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path

import pytest

from core.base_adapter import BaseAdapter


requires_api_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping live LLM test",
)


def _load_adapter_class_from_file(path: Path) -> type:
    """Import a standalone adapter.py and return its `BaseAdapter` subclass."""
    spec = importlib.util.spec_from_file_location("_test_generated_adapter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseAdapter)
            and obj is not BaseAdapter
        ):
            return obj
    raise AssertionError(f"No BaseAdapter subclass found in {path}")


@requires_api_key
def test_adapter_generator_produces_valid_tic_tac_toe_adapter(tmp_path: Path) -> None:
    """End-to-end: feed the generator a Tic-Tac-Toe description, verify the
    response is (a) syntactically valid Python and (b) survives the
    `validate_adapter` smoke test without an exception leaking out.

    Live test — uses the Anthropic API. Skipped automatically when
    ANTHROPIC_API_KEY is unset.
    """
    from llm.adapter_generator import AdapterGenerator

    description = (
        "Two-player Tic-Tac-Toe on a 3x3 grid. Players alternate placing their "
        "mark on any empty cell — player 0 plays X, player 1 plays O. The first "
        "player to align three of their own marks horizontally, vertically, or "
        "diagonally wins (+1 point, 0 for the loser). If the board fills up with "
        "no winner, both players score 0 and the game ends in a draw. "
        "Player 0 moves first. Actions identify the cell to mark by index "
        "0..8, row-major from the top-left."
    )

    generator = AdapterGenerator()
    path = generator.generate("tic_tac_toe", description, output_dir=tmp_path)

    assert path.exists() and path.is_file()
    code = path.read_text(encoding="utf-8")

    # Syntactic validity — generator already does this internally, but
    # making the assertion explicit guards against silent regressions.
    ast.parse(code)

    adapter_class = _load_adapter_class_from_file(path)
    assert issubclass(adapter_class, BaseAdapter)

    # validate_adapter must complete without raising, regardless of whether
    # the generated adapter is actually correct (some LLM outputs have bugs).
    report = generator.validate_adapter(adapter_class)
    assert isinstance(report, dict)
    assert {"ok", "instantiation", "games_attempted", "games_completed", "exceptions"} <= report.keys()
    assert report["instantiation"] != "not attempted"
