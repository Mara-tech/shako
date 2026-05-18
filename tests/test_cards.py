from __future__ import annotations

from core.engine import SimulationEngine
from games.cards.adapter import CardsAdapter
from rl.mcts_agent import MCTSAgent
from rl.random_agent import RandomAgent


def test_cards_game_runs_to_completion() -> None:
    """Sanity: random vs random plays a valid game with consistent final state."""
    adapter = CardsAdapter(seed=42)
    a, b = RandomAgent(seed=0), RandomAgent(seed=1)
    engine = SimulationEngine(adapter, [a, b])
    result = engine.run_game()

    # One turn per card played (2 * hand_size cards total).
    assert result.n_turns == 2 * adapter.hand_size
    # At most one point per trick; ties yield zero points for that trick.
    total = sum(result.scores.values())
    assert 0 <= total <= adapter.hand_size
    assert all(s >= 0 for s in result.scores.values())


def test_mcts_with_determinize_beats_random_on_cards() -> None:
    """With per-simulation determinization over the opponent's hidden hand,
    MCTS should comfortably outperform a uniformly random opponent.
    """
    adapter = CardsAdapter(seed=42)
    n_games = 500

    mcts = MCTSAgent(
        adapter,
        n_simulations=300,
        determinize=True,
        state_sampler=adapter.sample_state,
        seed=0,
    )
    rng_agent = RandomAgent(seed=1)
    engine = SimulationEngine(adapter, [mcts, rng_agent])

    wins = sum(1 for _ in range(n_games) if engine.run_game().winner_id == 0)
    rate = wins / n_games
    # This card game is heavily luck-dominated: 5 cards randomly drawn from 20,
    # with ties common (two copies of every value). A clean strict-winner rate
    # above 55% over 500 games is a comfortable demonstration that
    # determinization-driven MCTS extracts real edge over random play.
    assert rate > 0.55, f"MCTS won only {rate:.0%} of {n_games} games vs random"
