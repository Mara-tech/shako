from __future__ import annotations

from core.engine import SimulationEngine
from games.nim.adapter import NimAdapter
from games.nim.eval import nim_eval
from rl.greedy_agent import GreedyAgent
from rl.mcts_agent import MCTSAgent
from rl.random_agent import RandomAgent


def _win_rate(adapter: NimAdapter, agent_factory, opponent_factory, n_games: int) -> float:
    """Run `n_games` of agent-at-seat-0 vs opponent-at-seat-1, return win rate of seat 0."""
    a0 = agent_factory()
    a1 = opponent_factory()
    engine = SimulationEngine(adapter, [a0, a1])
    wins = sum(1 for _ in range(n_games) if engine.run_game().winner_id == 0)
    return wins / n_games


def test_mcts_beats_random_on_nim() -> None:
    """MCTS at seat 0 (the losing-position seat under misère 21-stick Nim)
    should still crush a uniformly random opponent.
    """
    adapter = NimAdapter(n_sticks=21, max_take=3, last_takes_wins=False)
    rate = _win_rate(
        adapter,
        agent_factory=lambda: MCTSAgent(adapter, n_simulations=500, seed=0),
        opponent_factory=lambda: RandomAgent(seed=1),
        n_games=100,
    )
    assert rate > 0.95, f"MCTS only won {rate:.0%} of 100 games vs random"


def test_mcts_approaches_optimal_greedy_winrate() -> None:
    """With enough simulations, MCTS should match optimal greedy's win rate
    against random play (both within a few percentage points of each other).
    """
    adapter = NimAdapter(n_sticks=21, max_take=3, last_takes_wins=False)
    n_games = 100

    greedy_rate = _win_rate(
        adapter,
        agent_factory=lambda: GreedyAgent(adapter, eval_fn=nim_eval, seed=0),
        opponent_factory=lambda: RandomAgent(seed=1),
        n_games=n_games,
    )
    mcts_rate = _win_rate(
        adapter,
        agent_factory=lambda: MCTSAgent(adapter, n_simulations=1000, seed=0),
        opponent_factory=lambda: RandomAgent(seed=1),
        n_games=n_games,
    )

    # Greedy is the upper bound (game-theoretic optimal play). MCTS with 1000
    # simulations on this tiny tree should be within 5pp on a 100-game sample.
    assert mcts_rate >= greedy_rate - 0.05, (
        f"MCTS {mcts_rate:.0%} lags greedy {greedy_rate:.0%} by too much"
    )
