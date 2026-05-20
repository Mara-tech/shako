from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any

from core.base_adapter import BaseAdapter
from core.types import Action, State
from rl.mcts_agent import MCTSAgent


def _state_key(state: State) -> str:
    """Canonical hashable key for a State. Sorted keys make `dict.data` deterministic."""
    return json.dumps(state.data, sort_keys=True, default=str)


def _action_key(action: Action) -> str:
    return json.dumps(action.data, sort_keys=True, default=str)


class PolicyMCTSAgent(MCTSAgent):
    """MCTS variant whose random rollouts are biased by an empirical policy table.

    The policy maps `(state_key, action_key) -> {"visits": int, "wins": int}` —
    typically aggregated from prior self-play games. During rollout, when the
    agent reaches a state with known statistics, it samples actions weighted by
    Laplace-smoothed win rates `(wins + 1) / (visits + 2)` rather than
    uniformly. Unseen state/action pairs fall back to a flat 0.5 weight, so the
    agent never refuses an action it hasn't seen before.

    With an empty policy it is behaviorally identical to `MCTSAgent`.
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        policy: dict[tuple[str, str], dict[str, int]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(adapter, **kwargs)
        self.policy: dict[tuple[str, str], dict[str, int]] = policy or {}

    def _rollout(self, state: State) -> dict[int, float]:
        state = self.adapter.clone_state(state)
        for _ in range(self.max_rollout_depth):
            if self.adapter.is_terminal(state):
                break
            pid = self.adapter.get_current_player(state)
            self.adapter.get_observable_state(state, pid)  # info-set discipline
            legal = self.adapter.get_legal_actions(state, pid)
            if not legal:
                break

            s_key = _state_key(state)
            weights: list[float] = []
            for action in legal:
                stats = self.policy.get((s_key, _action_key(action)))
                if stats is None:
                    weights.append(0.5)
                else:
                    weights.append((stats["wins"] + 1.0) / (stats["visits"] + 2.0))

            action = self._rng.choices(legal, weights=weights, k=1)[0]
            state = self.adapter.apply_action(state, action, pid)
        return self.adapter.get_scores(state)


class SelfPlayTrainer:
    """Iterative self-play loop that bootstraps a stronger MCTS agent.

    Each iteration:
      1. The current best agent plays `n_games_per_iter` games against itself.
      2. (state, action, winner) tuples are aggregated into a policy table.
      3. A candidate `PolicyMCTSAgent` is constructed with that table.
      4. Candidate plays `eval_games` against the current best, alternating
         seats to neutralize first-mover bias.
      5. If the candidate's win rate exceeds `promotion_threshold` (default
         0.55), it replaces the current best for the next iteration.

    Why MCTS policy biasing and not stable-baselines3:
      - The framework's `BaseAdapter` has variable action spaces; SB3 requires
        a fixed flat space wrapped in a Gym env, which is a substantial
        plumbing effort and a heavy dependency (torch, gymnasium).
      - Policy biasing keeps the approach framework-native and works on any
        adapter, including imperfect-information ones — no neural net needed
        for the small tabular games the framework currently targets.
      - It's the simplest improvement that demonstrably tightens MCTS's value
        estimates (strong rollouts beat random rollouts).

    The policy is rebuilt from scratch each iteration from the latest games
    only — older agent-current data would be off-policy for a newly-promoted
    candidate. Accumulating across iterations would need an AlphaZero-style
    rolling window; out of scope here.
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        n_iterations: int = 10,
        n_games_per_iter: int = 50,
        eval_games: int = 40,
        mcts_simulations: int = 100,
        max_turns: int = 1000,
        promotion_threshold: float = 0.55,
        verbose: bool = True,
        seed: int | None = None,
    ) -> None:
        self.adapter = adapter
        self.n_iterations = n_iterations
        self.n_games_per_iter = n_games_per_iter
        self.eval_games = eval_games
        self.mcts_simulations = mcts_simulations
        self.max_turns = max_turns
        self.promotion_threshold = promotion_threshold
        self.verbose = verbose
        self._rng = random.Random(seed)
        # Separate seed stream so derived-agent seeds don't intersect with
        # trainer-internal randomness (fallback illegal-action choice, etc.).
        self._agent_seeds = random.Random(seed)

        self.current_agent: MCTSAgent = MCTSAgent(
            adapter,
            n_simulations=mcts_simulations,
            seed=self._next_agent_seed(),
        )
        self.history: list[dict[str, Any]] = []

    def train(self) -> tuple[MCTSAgent, list[dict[str, Any]]]:
        """Run the full self-play schedule. Returns the best agent + per-iteration metrics."""
        for it in range(self.n_iterations):
            trajectories = self._self_play(self.current_agent, self.n_games_per_iter)
            policy = self._build_policy(trajectories)

            candidate = PolicyMCTSAgent(
                self.adapter,
                policy=policy,
                n_simulations=self.mcts_simulations,
                seed=self._next_agent_seed(),
            )

            win_rate = self._evaluate(candidate, self.current_agent, self.eval_games)
            promoted = win_rate > self.promotion_threshold
            if promoted:
                self.current_agent = candidate

            metrics = {
                "iteration": it,
                "self_play_games": self.n_games_per_iter,
                "policy_unique_states": len({k[0] for k in policy.keys()}),
                "policy_entries": len(policy),
                "candidate_win_rate": win_rate,
                "promoted": promoted,
            }
            self.history.append(metrics)
            if self.verbose:
                status = "promoted" if promoted else "rejected"
                print(
                    f"[iter {it:>2}] policy={metrics['policy_unique_states']:>5} states  "
                    f"candidate WR={win_rate:>6.2%}  {status}"
                )

        return self.current_agent, self.history

    def save_agent(self, path: str | Path) -> None:
        """Pickle the current best agent to `path`.

        The adapter, RNG state, and (for PolicyMCTSAgent) the policy table are
        all serialized. Loading requires `rl.self_play` to be importable.
        """
        with open(path, "wb") as f:
            pickle.dump(self.current_agent, f)

    def load_agent(self, path: str | Path) -> MCTSAgent:
        """Load a pickled agent, set it as `current_agent`, and return it."""
        with open(path, "rb") as f:
            self.current_agent = pickle.load(f)
        return self.current_agent

    # ------------------------------------------------------------------ internals

    def _next_agent_seed(self) -> int:
        return self._agent_seeds.randint(0, 2**31 - 1)

    def _self_play(
        self,
        agent: MCTSAgent,
        n_games: int,
    ) -> list[tuple[list[tuple[State, Action, int]], int | None]]:
        # Same instance plays both sides. RNG state evolves between games, so
        # successive games diverge naturally (important for fixed-initial-state
        # games like Nim where the deal is deterministic).
        return [self._play_one_game(agent, agent) for _ in range(n_games)]

    def _evaluate(self, candidate: MCTSAgent, current: MCTSAgent, n_games: int) -> float:
        wins = 0
        for i in range(n_games):
            if i % 2 == 0:
                _, winner = self._play_one_game(candidate, current)
                if winner == 0:
                    wins += 1
            else:
                _, winner = self._play_one_game(current, candidate)
                if winner == 1:
                    wins += 1
        return wins / n_games

    def _play_one_game(
        self,
        agent_a: MCTSAgent,
        agent_b: MCTSAgent,
    ) -> tuple[list[tuple[State, Action, int]], int | None]:
        n_players = self.adapter.get_n_players()
        # Even seats → agent_a, odd seats → agent_b.  For 2-player games this
        # is identical to the original [agent_a, agent_b] indexing.
        agents = [agent_a if pid % 2 == 0 else agent_b for pid in range(n_players)]
        for pid, agent in enumerate(agents):
            agent.on_game_start(pid, n_players)

        state = self.adapter.get_initial_state()
        trajectory: list[tuple[State, Action, int]] = []
        n_turns = 0
        timed_out = False

        while not self.adapter.is_terminal(state):
            if n_turns >= self.max_turns:
                timed_out = True
                break
            pid = self.adapter.get_current_player(state)
            legal = self.adapter.get_legal_actions(state, pid)
            if not legal:
                break
            obs = self.adapter.get_observable_state(state, pid)
            action = agents[pid].choose_action(obs, legal)
            if action not in legal:
                action = self._rng.choice(legal)
            # Adapter contract: apply_action does not mutate `state`, so the
            # reference we just appended stays a valid snapshot of this turn.
            trajectory.append((state, action, pid))
            state = self.adapter.apply_action(state, action, pid)
            n_turns += 1

        if timed_out:
            scores: dict[int, float] = {pid: 0.0 for pid in range(n_players)}
            winner: int | None = None
        else:
            scores = self.adapter.get_scores(state)
            top = max(scores.values())
            winners = [pid for pid, s in scores.items() if s == top]
            winner = winners[0] if len(winners) == 1 else None

        for agent in agents:
            agent.on_game_end(scores)

        return trajectory, winner

    @staticmethod
    def _build_policy(
        trajectories: list[tuple[list[tuple[State, Action, int]], int | None]],
    ) -> dict[tuple[str, str], dict[str, int]]:
        policy: dict[tuple[str, str], dict[str, int]] = {}
        for trajectory, winner in trajectories:
            for state, action, pid in trajectory:
                key = (_state_key(state), _action_key(action))
                if key not in policy:
                    policy[key] = {"visits": 0, "wins": 0}
                policy[key]["visits"] += 1
                if winner is not None and pid == winner:
                    policy[key]["wins"] += 1
        return policy
