from __future__ import annotations

import math
import random
from typing import Callable

from core.base_adapter import BaseAdapter
from core.base_agent import BaseAgent
from core.types import Action, ObservableState, State


EvalFn = Callable[[ObservableState], float]


class GreedyAgent(BaseAgent):
    """One-step lookahead agent.

    For each legal action, the agent clones the current state, applies the
    action via the adapter, and scores the resulting observable state with
    `eval_fn`. The highest-scoring action is chosen (ties broken at random).

    Without an `eval_fn`, the agent degenerates into a `RandomAgent`.

    Important caveat for imperfect-information games: the agent only has access
    to `observable_state`, not the true `state`. It treats the observable state
    as its best guess at the full state and simulates from there. This is exact
    for perfect-information games (chess, go, ...) and a reasonable heuristic
    elsewhere, but the adapter must accept partial states without raising.
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        eval_fn: EvalFn | None = None,
        seed: int | None = None,
    ) -> None:
        """Args:
            adapter: the game adapter, used to clone and apply actions during lookahead.
            eval_fn: scoring function applied to the post-action observable state.
                `None` means "behave as a `RandomAgent`".
            seed: RNG seed used for random fallback and tiebreaking.
        """
        self.adapter = adapter
        self.eval_fn = eval_fn
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
        if self.eval_fn is None:
            return self._rng.choice(legal_actions)

        pid = observable_state.player_id
        pseudo_state = State(data=dict(observable_state.data))

        best_score = -math.inf
        best_actions: list[Action] = []
        for action in legal_actions:
            clone = self.adapter.clone_state(pseudo_state)
            next_state = self.adapter.apply_action(clone, action, pid)
            next_obs = self.adapter.get_observable_state(next_state, pid)
            score = self.eval_fn(next_obs)
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return self._rng.choice(best_actions)
