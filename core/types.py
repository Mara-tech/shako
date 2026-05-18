from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class State:
    """Full game state, including hidden information. Only the adapter should manipulate this directly."""

    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservableState:
    """Partial view of the game state visible to a specific player."""

    data: dict[str, Any] = field(default_factory=dict)
    player_id: int = 0


@dataclass
class Action:
    """A player action. The `data` dict carries game-specific payload."""

    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class GameResult:
    """Summary produced at the end of a completed game."""

    scores: dict[int, float]
    n_turns: int
    winner_id: int | None
    duration_ms: float
    illegal_action_counts: dict[int, int] = field(default_factory=dict)
    actions: list[tuple[int, int, Action]] | None = None
    timed_out: bool = False
