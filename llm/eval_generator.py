from __future__ import annotations

import ast
import re
from pathlib import Path

import anthropic


_ROOT = Path(__file__).resolve().parent.parent
_TYPES_SRC = (_ROOT / "core" / "types.py").read_text(encoding="utf-8")
_NIM_EVAL_SRC = (_ROOT / "games" / "nim" / "eval.py").read_text(encoding="utf-8")


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


class EvalGenerator:
    """LLM-driven evaluation-function generator.

    Companion to `AdapterGenerator`. Produces a `<game_name>_eval` function
    that scores an `ObservableState` from the perspective of "is this position
    bad for the player about to move". Higher = worse for the mover, so
    `GreedyAgent` (which maximizes `eval_fn` on the post-action state) plays
    toward states that are good for itself.

    If `games/<game_name>/adapter.py` exists, its source is included in the
    user message so the model knows exactly which keys live in `state.data`.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 8_000,
    ) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self._system_prompt = self._build_system_prompt()

    def generate(
        self,
        game_name: str,
        criteria: str,
        output_dir: Path | str = _ROOT / "games",
    ) -> Path:
        """Generate an eval function and write it to `<output_dir>/<game_name>/eval.py`.

        Args:
            game_name: snake_case folder name; the function is `<game_name>_eval`.
            criteria: free-text description of what makes a player advantaged
                (which features of the state correlate with winning).
            output_dir: parent directory of the game folder.

        Returns:
            Path to the written eval file.
        """
        response_text = self._call_claude(game_name, criteria, output_dir)
        code = self._extract_python(response_text)
        self._validate_syntax(code)

        target = Path(output_dir) / game_name / "eval.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")
        init_file = target.parent / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")
        return target

    # ------------------------------------------------------------------ internals

    def _call_claude(self, game_name: str, criteria: str, output_dir: Path | str) -> str:
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
                    "content": self._build_user_message(game_name, criteria, output_dir),
                }
            ],
        )
        return next(b.text for b in message.content if b.type == "text")

    def _build_user_message(
        self,
        game_name: str,
        criteria: str,
        output_dir: Path | str,
    ) -> str:
        fn_name = f"{game_name}_eval"
        adapter_path = Path(output_dir) / game_name / "adapter.py"
        adapter_section = ""
        if adapter_path.exists():
            adapter_src = adapter_path.read_text(encoding="utf-8")
            adapter_section = (
                f"\nThe adapter for this game is already written. Read it to learn "
                f"the exact shape of `state.data` your eval must inspect:\n\n"
                f"```python\n{adapter_src}\n```\n"
            )

        return (
            f"Function name (mandatory): {fn_name}\n"
            f"Signature: `def {fn_name}(state: ObservableState) -> float`\n"
            f"{adapter_section}\n"
            f"Advantage criteria — what makes a player better positioned in this game:\n"
            f"\"\"\"\n{criteria.strip()}\n\"\"\"\n\n"
            f"Return only the complete Python file in a single ```python code block."
        )

    def _build_system_prompt(self) -> str:
        return f"""You are generating Python evaluation functions for the shako framework.

# Goal

Produce a function `<game_name>_eval(state: ObservableState) -> float` that scores a position from the perspective of "is this bad for the player about to move".

Higher score = WORSE for the player to move (= better for the player who just acted to reach this state). `GreedyAgent` will apply each candidate action, then maximize this eval on the resulting `ObservableState` — so the eval should reward states that are losing for the *opponent* about to respond.

# Shared types

```python
{_TYPES_SRC}
```

# Conventions

- The function reads `state.data` (a dict) and `state.player_id` if needed.
- Return a finite float. Magnitudes don't matter as long as ordering is correct.
- Pure function — no side effects, no I/O.
- If the game has multiple advantage signals (material, position, tempo), combine them as a weighted sum and document the weights in a one-line comment.

# Required imports

Your file MUST start with:

```python
from __future__ import annotations

from core.types import ObservableState
```

Add `from typing import Callable` and a factory function if the game variant has tunable parameters.

# Reference example: Nim eval

A working eval for misère Nim showing the convention (closed-form "is the resulting position losing for the player to move").

```python
{_NIM_EVAL_SRC}
```

# Output format

Return EXACTLY ONE Python code block, fenced with ```python ... ```. No prose before or after.
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
