from __future__ import annotations

import pytest

from core.engine import SimulationEngine
from core.types import State
from games.nim.adapter import NimAdapter
from games.nim.eval import make_nim_eval, nim_eval
from rl.greedy_agent import GreedyAgent
from rl.random_agent import RandomAgent


# -------- legal actions --------------------------------------------------------


def test_legal_actions_full_pile() -> None:
    adapter = NimAdapter(n_sticks=21, max_take=3)
    state = adapter.get_initial_state()
    legal = adapter.get_legal_actions(state, player_id=0)
    assert [a.data["take"] for a in legal] == [1, 2, 3]


def test_legal_actions_capped_by_remaining_sticks() -> None:
    adapter = NimAdapter(n_sticks=21, max_take=3)

    state_two = State(data={"sticks": 2, "current_player": 0})
    assert [a.data["take"] for a in adapter.get_legal_actions(state_two, 0)] == [1, 2]

    state_one = State(data={"sticks": 1, "current_player": 1})
    assert [a.data["take"] for a in adapter.get_legal_actions(state_one, 1)] == [1]


def test_legal_actions_with_custom_max_take() -> None:
    adapter = NimAdapter(n_sticks=10, max_take=5)
    state = adapter.get_initial_state()
    legal = adapter.get_legal_actions(state, 0)
    assert [a.data["take"] for a in legal] == [1, 2, 3, 4, 5]


def test_terminal_and_scores_misere() -> None:
    adapter = NimAdapter(last_takes_wins=False)
    # Player 0 just took the last stick -> current_player flips to 1.
    terminal = State(data={"sticks": 0, "current_player": 1})
    assert adapter.is_terminal(terminal)
    assert adapter.get_scores(terminal) == {0: 0.0, 1: 1.0}


def test_terminal_and_scores_normal_play() -> None:
    adapter = NimAdapter(last_takes_wins=True)
    terminal = State(data={"sticks": 0, "current_player": 1})
    assert adapter.get_scores(terminal) == {0: 1.0, 1: 0.0}


# -------- greedy vs random -----------------------------------------------------


def test_greedy_with_nim_eval_beats_random() -> None:
    """Greedy plays seat 0 from the misère losing position (21 sticks).

    Even starting in a theoretically lost position, GreedyAgent crushes
    RandomAgent because random almost never punishes greedy's forced
    random moves — so the game keeps drifting back into greedy's favor.
    """
    adapter = NimAdapter(n_sticks=21, max_take=3, last_takes_wins=False)
    greedy = GreedyAgent(adapter, eval_fn=nim_eval, seed=0)
    rng_agent = RandomAgent(seed=1)
    engine = SimulationEngine(adapter, [greedy, rng_agent])

    n_games = 1000
    greedy_wins = sum(1 for _ in range(n_games) if engine.run_game().winner_id == 0)
    assert greedy_wins / n_games > 0.9, f"greedy won only {greedy_wins}/{n_games}"


# -------- optimal vs optimal: deterministic winner -----------------------------


def test_two_optimal_greedy_agents_produce_deterministic_winner() -> None:
    """Both seats play optimally; whoever starts in a winning position must win.

    With n_sticks=21 (misère, 21 % 4 == 1), seat 0 is in a losing position
    from the very first move. Optimal play from seat 1 should win every
    single game regardless of the RNG seeds used for tiebreaking.
    """
    adapter = NimAdapter(n_sticks=21, max_take=3, last_takes_wins=False)
    winners: set[int | None] = set()
    for seed in range(25):
        a0 = GreedyAgent(adapter, eval_fn=nim_eval, seed=seed)
        a1 = GreedyAgent(adapter, eval_fn=nim_eval, seed=seed + 10_000)
        engine = SimulationEngine(adapter, [a0, a1])
        winners.add(engine.run_game().winner_id)
    assert winners == {1}, f"non-deterministic winner across seeds: {winners}"


# -------- bonus: factory eval works for non-default variants -------------------


@pytest.mark.parametrize("n_sticks", [16, 20, 24])
def test_make_nim_eval_winning_seat_wins_normal_play(n_sticks: int) -> None:
    """Normal-play variant: losing positions are multiples of (max_take+1).

    With n_sticks divisible by 4, seat 0 is losing → seat 1 wins every game.
    """
    adapter = NimAdapter(n_sticks=n_sticks, max_take=3, last_takes_wins=True)
    eval_fn = make_nim_eval(max_take=3, last_takes_wins=True)
    a0 = GreedyAgent(adapter, eval_fn=eval_fn, seed=0)
    a1 = GreedyAgent(adapter, eval_fn=eval_fn, seed=1)
    engine = SimulationEngine(adapter, [a0, a1])
    winners = {engine.run_game().winner_id for _ in range(10)}
    assert winners == {1}
