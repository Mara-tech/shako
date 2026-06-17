from __future__ import annotations

import copy
from abc import ABC, abstractmethod

from core.types import Action, ObservableState, State


class BaseAdapter(ABC):
    """Abstract interface every game must implement to be used by the balancer."""

    @abstractmethod
    def get_initial_state(self) -> State:
        """Return a fresh state representing the start of a new game.

        Called once by the engine at the beginning of every `run_game`.
        """

    @abstractmethod
    def get_legal_actions(self, state: State, player_id: int) -> list[Action]:
        """Return all actions that `player_id` may legally take in `state`.

        The returned list must be non-empty for any non-terminal state where
        `get_current_player` returns `player_id`.
        """

    @abstractmethod
    def apply_action(self, state: State, action: Action, player_id: int) -> State:
        """Return the new state that results from `player_id` applying `action`.

        Must not mutate `state`; return a fresh (or cloned-then-mutated) State.
        """

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """Return True if the game is over and no more actions can be taken."""

    @abstractmethod
    def get_scores(self, state: State) -> dict[int, float]:
        """Return the final (or current) score for every player.

        Keys are player ids (0-indexed). Should only be called on a terminal
        state for meaningful results, but implementations may return
        intermediate scores when called mid-game.
        """

    @abstractmethod
    def get_current_player(self, state: State) -> int:
        """Return the id of the player who must act next in `state`."""

    @abstractmethod
    def get_observable_state(self, state: State, player_id: int) -> ObservableState:
        """Return the part of `state` that `player_id` is allowed to see.

        Hidden information (e.g. opponents' hands, face-down cards) must be
        stripped. The returned ObservableState should be self-contained enough
        for an agent to choose an action.
        """

    @abstractmethod
    def get_n_players(self) -> int:
        """Return the fixed number of players for this game variant."""

    @abstractmethod
    def clone_state(self, state: State) -> State:
        """Return a deep copy of `state`.

        Used heavily by MCTS to simulate rollouts without corrupting the real
        game tree. Implementations may override with a faster custom copy if
        the default `copy.deepcopy` is too slow.
        """
        return copy.deepcopy(state)  # default fallback; subclasses may override

    def get_rich_renderable(self, obs_state: ObservableState):
        """Return a Rich ``RenderableType`` for display in the Rich/Textual human UI.

        Override to show a visual board instead of a raw dict.
        Return ``None`` (default) to fall back to dict display.
        This method is intentionally non-abstract: adapters opt in at their own pace.
        """
        return None

    def get_action_label(self, action: Action) -> str:
        """Coarse label for `action`, used by the analyzer for frequency analysis.

        Override to group semantically equivalent actions (e.g. "play_2_cards"
        instead of the full card list). The default serialises action.data and
        is correct for simple games. Overriding avoids false-positive rare-action
        reports in games with large combinatorial action spaces.
        """
        import json
        return json.dumps(action.data, sort_keys=True, default=str)
