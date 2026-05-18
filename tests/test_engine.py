from __future__ import annotations

from core.base_agent import BaseAgent
from core.engine import SimulationEngine
from core.types import Action, ObservableState
from games.nim.adapter import NimAdapter
from rl.random_agent import RandomAgent


class _AlwaysIllegalAgent(BaseAgent):
    """Returns a sentinel action that is never in any legal-action list.

    Module-level (not nested) so multiprocessing-pickling sees it if a future
    test puts it through `run_batch`.
    """

    def on_game_start(self, player_id: int, n_players: int) -> None:
        pass

    def on_game_end(self, scores: dict[int, float]) -> None:
        pass

    def choose_action(self, observable_state: ObservableState, legal_actions: list[Action]) -> Action:
        return Action(data={"__never_legal__": True})


def test_run_batch_parallel_completes_every_game() -> None:
    """`run_batch(n_workers > 1)` spawns processes via multiprocessing and
    returns one GameResult per game, all finished cleanly.

    Kept small to bound the Windows spawn-startup cost.
    """
    adapter = NimAdapter(n_sticks=11, max_take=3)
    agents = [RandomAgent(seed=0), RandomAgent(seed=1)]
    engine = SimulationEngine(adapter, agents)

    results = engine.run_batch(n_games=6, n_workers=2)
    assert len(results) == 6
    for r in results:
        assert r.n_turns > 0
        assert not r.timed_out
        assert sum(r.scores.values()) == 1.0  # exactly one winner per Nim game


def test_seeded_random_agents_produce_identical_games() -> None:
    """Two engines built with identically-seeded agents and adapter must
    produce byte-identical results across replays.
    """
    def build_engine() -> SimulationEngine:
        adapter = NimAdapter(n_sticks=15, max_take=3)
        agents = [RandomAgent(seed=42), RandomAgent(seed=43)]
        return SimulationEngine(adapter, agents, seed=0)

    r1 = [build_engine().run_game() for _ in range(3)]
    r2 = [build_engine().run_game() for _ in range(3)]

    for a, b in zip(r1, r2):
        assert a.scores == b.scores
        assert a.n_turns == b.n_turns
        assert a.winner_id == b.winner_id


def test_illegal_action_is_replaced_by_random_and_counted() -> None:
    """When an agent returns an action absent from the legal list, the engine
    must (a) substitute a uniformly random legal action, (b) bump
    `illegal_action_counts[player_id]`, and (c) still complete the game.
    """
    adapter = NimAdapter(n_sticks=11, max_take=3)
    agents = [_AlwaysIllegalAgent(), RandomAgent(seed=0)]
    engine = SimulationEngine(adapter, agents, seed=0)

    result = engine.run_game()
    assert not result.timed_out
    # Player 0 played at least one turn — Nim never lets a single player skip —
    # and every one of those plays was illegal.
    bad_count = result.illegal_action_counts.get(0, 0)
    assert bad_count > 0
    # Player 1 always plays a legal action.
    assert result.illegal_action_counts.get(1, 0) == 0
