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

    def on_state_update(self, observable_state: ObservableState) -> None:
        """Optional hook called after every applied action, whoever's turn it was.

        Unlike `choose_action`, this fires even on turns this agent isn't
        acting on — including the terminal one. Override for agents backing a
        live display (e.g. a UI) that must stay in sync while another player
        or a slow bot is thinking. Default is a no-op; the engine skips the
        (state, hook) bookkeeping entirely for agents that don't override it.
        """
        return None
