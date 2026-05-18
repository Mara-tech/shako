from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import anthropic

from core.base_adapter import BaseAdapter
from core.engine import SimulationEngine
from rl.random_agent import RandomAgent


_ROOT = Path(__file__).resolve().parent.parent
_BASE_ADAPTER_SRC = (_ROOT / "core" / "base_adapter.py").read_text(encoding="utf-8")
_TYPES_SRC = (_ROOT / "core" / "types.py").read_text(encoding="utf-8")
_NIM_ADAPTER_SRC = (_ROOT / "games" / "nim" / "adapter.py").read_text(encoding="utf-8")


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _class_name(game_name: str) -> str:
    """Convert snake_case (or kebab-case) game name to <CamelCase>Adapter."""
    parts = re.split(r"[_\-]+", game_name)
    return "".join(p.capitalize() for p in parts if p) + "Adapter"


class AdapterGenerator:
    """LLM-driven adapter generator.

    Builds a stable, fully-specified system prompt (interface + types + Nim
    reference) and asks Claude to fill in `apply_action`-and-friends for a new
    game. Prompt caching is enabled on the system block so successive calls
    only pay the cache-read price for the framework reference (~0.1x).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 16_000,
    ) -> None:
        """Args:
            api_key: Anthropic API key. Falls back to `ANTHROPIC_API_KEY` env var.
            model: Claude model ID. Sonnet 4.6 is the sweet spot for codegen quality vs. cost.
            max_tokens: response cap; typical adapters fit well under 4k tokens.
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self._system_prompt = self._build_system_prompt()

    def generate(
        self,
        game_name: str,
        description: str,
        output_dir: Path | str = _ROOT / "games",
    ) -> Path:
        """Generate an adapter for `description` and write it to disk.

        Args:
            game_name: snake_case folder name; the class is named `<CamelCase>Adapter`.
            description: free-text rules for the game.
            output_dir: parent directory; the file lands at `<output_dir>/<game_name>/adapter.py`.

        Returns:
            Path to the written adapter file.

        Raises:
            ValueError: if Claude's response contains no Python code block, or
                the extracted code fails `ast.parse`.
        """
        response_text = self._call_claude(game_name, description)
        code = self._extract_python(response_text)
        self._validate_syntax(code)

        target = Path(output_dir) / game_name / "adapter.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")
        # Ensure the package is importable.
        init_file = target.parent / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")
        return target

    def validate_adapter(self, adapter_class: type) -> dict[str, Any]:
        """Smoke-test a generated adapter by running 10 random-vs-random games.

        The report distinguishes three failure modes:
        - Construction failure (missing required ctor arg, bad inheritance)
        - Per-game exception (bug in apply_action / get_legal_actions / etc.)
        - Non-termination (no legal-action path leads to a terminal state)

        Returns:
            {
              "ok": bool,                          # True iff all 10 games finished cleanly
              "instantiation": str,                # "ok" or "failed: ..."
              "is_base_adapter": bool,
              "n_players": int | None,
              "games_attempted": 10,
              "games_completed": int,
              "exceptions": list[str],
            }
        """
        report: dict[str, Any] = {
            "ok": False,
            "instantiation": "not attempted",
            "is_base_adapter": False,
            "n_players": None,
            "games_attempted": 10,
            "games_completed": 0,
            "exceptions": [],
        }

        try:
            adapter = adapter_class()
        except Exception as e:
            report["instantiation"] = f"failed: {type(e).__name__}: {e}"
            return report
        report["instantiation"] = "ok"
        report["is_base_adapter"] = isinstance(adapter, BaseAdapter)
        if not report["is_base_adapter"]:
            report["exceptions"].append(
                f"{adapter_class.__name__} does not subclass BaseAdapter"
            )
            return report

        try:
            n_players = adapter.get_n_players()
            report["n_players"] = n_players
        except Exception as e:
            report["exceptions"].append(f"get_n_players(): {type(e).__name__}: {e}")
            return report

        agents = [RandomAgent(seed=i) for i in range(n_players)]
        engine = SimulationEngine(adapter, agents, max_turns=1000)
        for i in range(10):
            try:
                result = engine.run_game()
                if result.timed_out:
                    report["exceptions"].append(f"game {i}: hit max_turns (1000) without terminating")
                else:
                    report["games_completed"] += 1
            except Exception as e:
                report["exceptions"].append(f"game {i}: {type(e).__name__}: {e}")

        report["ok"] = report["games_completed"] == report["games_attempted"]
        return report

    # ------------------------------------------------------------------ internals

    def _call_claude(self, game_name: str, description: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": self._system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": self._build_user_message(game_name, description),
                }
            ],
        )
        return next(b.text for b in message.content if b.type == "text")

    def _build_user_message(self, game_name: str, description: str) -> str:
        cls = _class_name(game_name)
        return (
            f"Implement the adapter for the following game.\n\n"
            f"Class name (mandatory): {cls}\n\n"
            f"Game description:\n\"\"\"\n{description.strip()}\n\"\"\"\n\n"
            f"Return only the complete Python file in a single ```python code block."
        )

    def _build_system_prompt(self) -> str:
        return f"""You are generating Python game adapters for the shako framework.

# Interface to implement (BaseAdapter)

The adapter MUST subclass `BaseAdapter` and implement every abstract method.

```python
{_BASE_ADAPTER_SRC}
```

# Shared types

```python
{_TYPES_SRC}
```

# Conventions

- `State.data` is a free-form dict carrying the full game state, INCLUDING hidden information.
- `ObservableState.data` is the partial view visible to one player; hidden information MUST be stripped.
- `Action.data` is a free-form dict describing one move (e.g. `{{"card": 5}}`).
- `clone_state` MUST produce a deep copy. Lists/dicts inside `state.data` must not be shared with the original.
- `apply_action` MUST NOT mutate the input state — clone first, then mutate the clone.
- `get_legal_actions` must return a non-empty list whenever the state is non-terminal and it's the queried player's turn.
- Players are 0-indexed; `get_n_players()` returns the count.

# Cases to handle

- **Hidden information**: implement `get_observable_state` to strip what `player_id` can't see (opponent hands, face-down cards, ...).
- **Randomness**: keep an RNG seeded in `__init__`; expose a `seed: int | None = None` parameter.
- **Multi-player**: support 2+ players where the rules call for it.

# Required imports

Your file MUST start with:

```python
from __future__ import annotations

from core.base_adapter import BaseAdapter
from core.types import Action, ObservableState, State
```

Add `import random` (or other stdlib only) if you need randomness or copying.

# Reference example: Nim adapter

This is a working adapter using the same conventions you must follow.

```python
{_NIM_ADAPTER_SRC}
```

# Output format

Return EXACTLY ONE Python code block, fenced with ```python ... ```. The block must contain the complete adapter file, ready to drop into `games/<game_name>/adapter.py`. No prose before or after.
"""

    @staticmethod
    def _extract_python(response: str) -> str:
        match = _CODE_BLOCK_RE.search(response)
        if not match:
            raise ValueError("No Python code block found in model response")
        return match.group(1).rstrip() + "\n"

    @staticmethod
    def _validate_syntax(code: str) -> None:
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Generated code failed to parse: {e}") from e
