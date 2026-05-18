from __future__ import annotations

import random

from core.base_agent import BaseAgent
from core.types import Action, ObservableState


class RandomAgent(BaseAgent):
    """Picks a uniformly random legal action every turn.

    Useful as a baseline opponent and as a safe fallback when a more complex
    agent fails or times out.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Args:
            seed: RNG seed for reproducible play. `None` uses fresh randomness.
        """
        self._rng = random.Random(seed)
        self.player_id: int = 0

    def on_game_start(self, player_id: int, n_players: int) -> None:
        self.player_id = player_id

    def on_game_end(self, scores: dict[int, float]) -> None:
        pass

    def choose_action(
        self,
        observable_state: ObservableState,
        legal_actions: list[Action],
    ) -> Action:
        return self._rng.choice(legal_actions)
