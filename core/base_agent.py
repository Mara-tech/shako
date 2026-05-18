from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import Action, ObservableState


class BaseAgent(ABC):
    """Abstract interface for any agent (human proxy, RL policy, LLM, random…)."""

    @abstractmethod
    def choose_action(
        self,
        observable_state: ObservableState,
        legal_actions: list[Action],
    ) -> Action:
        """Select and return one action from `legal_actions`.

        Args:
            observable_state: The portion of the game state visible to this agent.
            legal_actions: Non-empty list of actions the agent is allowed to take.

        Returns:
            The chosen action. Must be an element of `legal_actions`.
        """

    @abstractmethod
    def on_game_start(self, player_id: int, n_players: int) -> None:
        """Called once before the first turn of every new game.

        Args:
            player_id: The seat this agent occupies in the current game.
            n_players: Total number of players in the game.
        """

    @abstractmethod
    def on_game_end(self, scores: dict[int, float]) -> None:
        """Called once after the terminal state is reached.

        Agents that learn online (e.g. RL agents) should update their policy
        here using the final `scores`.

        Args:
            scores: Mapping from player_id to final score for every player.
        """
