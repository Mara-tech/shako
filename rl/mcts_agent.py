from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable

from core.base_adapter import BaseAdapter
from core.base_agent import BaseAgent
from core.types import Action, ObservableState, State


StateSampler = Callable[[ObservableState], State]


@dataclass
class _Node:
    """A node in the MCTS search tree.

    Notes:
      * The full game state is NOT cached here — for imperfect-information
        play with per-simulation determinization, the state at this node
        depends on which world was sampled. We recompute it by replaying
        `action_from_parent` from the (freshly sampled) root each sim.
      * `value_sum` is accumulated from the perspective of the player who
        chose this node (i.e. `parent.to_move`). UCB selection at the parent
        therefore directly maximizes that player's expected score.
      * `to_move` is the player whose turn it is at this node. We assume it
        is a deterministic function of the action path from root, not of
        hidden information — true for essentially all turn-based games.
    """

    parent: "_Node | None" = None
    action_from_parent: Action | None = None
    children: list["_Node"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    to_move: int = -1  # -1 means "terminal / unknown"


class MCTSAgent(BaseAgent):
    """UCT-based Monte-Carlo Tree Search agent.

    Works with any `BaseAdapter` and requires no evaluation function — uses
    uniform random rollouts to estimate state values. With enough simulations,
    converges toward the game-theoretic optimum.

    Imperfect-information games: enable `determinize=True` and (optionally)
    pass a `state_sampler` that draws a plausible full state from an observable
    state. Each simulation re-samples the world, so the tree statistics average
    over the agent's information set rather than committing to a single
    hypothesis. This is the "Perfect Information MCTS" (PIMC) extension of
    standard UCT — not full IS-MCTS, but a substantial improvement over plain
    determinization-once-per-move.
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        n_simulations: int = 500,
        max_rollout_depth: int = 50,
        c_exploration: float = 1.41,
        time_limit_ms: int | None = None,
        determinize: bool = False,
        state_sampler: StateSampler | None = None,
        seed: int | None = None,
    ) -> None:
        """Args:
            adapter: the game adapter.
            n_simulations: number of MCTS rollouts per `choose_action` call.
                Ignored if `time_limit_ms` is set.
            max_rollout_depth: hard cap on the depth of the random playout
                phase. Reaching the cap returns the current adapter scores.
            c_exploration: UCB1 exploration constant. The classic √2 ≈ 1.41
                is appropriate when rewards are normalized to [0, 1].
            time_limit_ms: if set, run as many simulations as fit in this
                wall-clock budget (per call) and ignore `n_simulations`.
            determinize: if True, sample a fresh full state at the start of
                every simulation. Required for sound play in games with
                hidden information.
            state_sampler: callable that turns an `ObservableState` into a
                plausible full `State`. If `determinize=True` and this is
                None, falls back to treating the observable state as full
                state (which is exact for perfect-info games).
            seed: RNG seed for rollouts, expansion, and tiebreaks.
        """
        self.adapter = adapter
        self.n_simulations = n_simulations
        self.max_rollout_depth = max_rollout_depth
        self.c_exploration = c_exploration
        self.time_limit_ms = time_limit_ms
        self.determinize = determinize
        self.state_sampler = state_sampler
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
        root = _Node(to_move=observable_state.player_id)

        if self.time_limit_ms is not None:
            deadline = time.perf_counter() + self.time_limit_ms / 1000.0
            while time.perf_counter() < deadline:
                self._run_one_simulation(root, observable_state)
        else:
            for _ in range(self.n_simulations):
                self._run_one_simulation(root, observable_state)

        if not root.children:
            return self._rng.choice(legal_actions)

        # Pick the action explored most often — by convention, more visits
        # under UCT means a more confident estimate of value.
        best = max(root.children, key=lambda c: c.visits)
        chosen = best.action_from_parent
        if chosen is None or chosen not in legal_actions:
            # Defensive fallback: determinization may have produced an action
            # set that doesn't intersect the engine's `legal_actions` for the
            # true state. Pick the most-visited child whose action IS legal,
            # else random.
            legal_children = [c for c in root.children if c.action_from_parent in legal_actions]
            if legal_children:
                best = max(legal_children, key=lambda c: c.visits)
                return best.action_from_parent  # type: ignore[return-value]
            return self._rng.choice(legal_actions)
        return chosen

    # ------------------------------------------------------------------ MCTS core

    def _run_one_simulation(self, root: _Node, observable_state: ObservableState) -> None:
        """One iteration of Selection → Expansion → Rollout → Backprop."""
        # Fresh determinization (or identity) per simulation.
        state = self._sample_root_state(observable_state)
        path: list[_Node] = [root]
        node = root

        # ---- Selection: descend while fully expanded under this determinization
        while True:
            if self.adapter.is_terminal(state):
                break
            legal = self.adapter.get_legal_actions(state, node.to_move)
            if not legal:
                break
            tried = [c.action_from_parent for c in node.children]
            if any(a not in tried for a in legal):
                break  # node has untried actions — stop and expand here
            # Only children whose action is legal in THIS determinization are eligible.
            eligible = [c for c in node.children if c.action_from_parent in legal]
            if not eligible:
                break
            parent_to_move = node.to_move
            node = self._ucb_select(node, eligible)
            state = self.adapter.apply_action(state, node.action_from_parent, parent_to_move)  # type: ignore[arg-type]
            path.append(node)

        # ---- Expansion: add one new child if room remains
        if not self.adapter.is_terminal(state):
            legal = self.adapter.get_legal_actions(state, node.to_move)
            tried = [c.action_from_parent for c in node.children]
            untried = [a for a in legal if a not in tried]
            if untried:
                action = self._rng.choice(untried)
                parent_to_move = node.to_move
                state = self.adapter.apply_action(state, action, parent_to_move)
                new_node = _Node(parent=node, action_from_parent=action)
                if not self.adapter.is_terminal(state):
                    new_node.to_move = self.adapter.get_current_player(state)
                node.children.append(new_node)
                path.append(new_node)
                node = new_node

        # ---- Rollout: random playout to terminal or depth cap
        scores = self._rollout(state)

        # ---- Backprop: update visits everywhere, value_sum from chooser's POV
        for i, n in enumerate(path):
            n.visits += 1
            if i > 0:
                chooser = path[i - 1].to_move
                n.value_sum += scores.get(chooser, 0.0)

    def _ucb_select(self, parent: _Node, children: list[_Node]) -> _Node:
        """Pick the child maximizing UCB1 from `parent.to_move`'s perspective."""
        log_parent = math.log(parent.visits) if parent.visits > 0 else 0.0
        best: _Node | None = None
        best_score = -math.inf
        for child in children:
            # Children reach this branch only after expansion, so visits >= 1.
            exploit = child.value_sum / child.visits
            explore = self.c_exploration * math.sqrt(log_parent / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best = child
        assert best is not None  # children list is non-empty by caller contract
        return best

    def _rollout(self, state: State) -> dict[int, float]:
        """Uniform random playout to terminal or `max_rollout_depth`.

        Calls `get_observable_state()` at every step so the adapter sees an
        information-set-respecting trace, even though random play doesn't
        actually use it. This keeps rollout discipline consistent with smarter
        playout policies we may plug in later.
        """
        state = self.adapter.clone_state(state)
        for _ in range(self.max_rollout_depth):
            if self.adapter.is_terminal(state):
                break
            pid = self.adapter.get_current_player(state)
            self.adapter.get_observable_state(state, pid)  # info-set discipline
            legal = self.adapter.get_legal_actions(state, pid)
            if not legal:
                break
            action = self._rng.choice(legal)
            state = self.adapter.apply_action(state, action, pid)
        return self.adapter.get_scores(state)

    def _sample_root_state(self, observable_state: ObservableState) -> State:
        if self.determinize and self.state_sampler is not None:
            return self.state_sampler(observable_state)
        return State(data=dict(observable_state.data))
