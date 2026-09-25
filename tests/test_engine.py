from __future__ import annotations

import pytest

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


class _RaisingNimAdapter(NimAdapter):
    """Nim whose `apply_action` refuses the move — the shape of a game that raises on a move."""

    def apply_action(self, state, action, player_id):  # type: ignore[no-untyped-def]
        raise RuntimeError("refused by the game")


def test_on_action_applied_sees_every_applied_action_in_order() -> None:
    """One call per applied action, in order, with the states on either side of it.

    Compared with `GameResult.actions`, which records the same actions from inside the loop:
    the hook is the live counterpart of `record=True`.
    """
    calls: list[tuple] = []
    adapter = NimAdapter(n_sticks=11, max_take=3)
    engine = SimulationEngine(
        adapter,
        [RandomAgent(seed=0), RandomAgent(seed=1)],
        record=True,
        seed=0,
        on_action_applied=lambda *args: calls.append(args),
    )
    result = engine.run_game()
    assert result.actions is not None

    assert [(pid, action) for _, action, pid, _ in calls] == [
        (pid, action) for _, pid, action in result.actions
    ]
    # Each call starts where the previous one ended, from the initial state to a terminal one.
    assert calls[0][0] == adapter.get_initial_state()
    for (_, _, _, reached), (following, _, _, _) in zip(calls, calls[1:]):
        assert following == reached
    assert adapter.is_terminal(calls[-1][3])
    for state, action, pid, reached in calls:
        assert reached == adapter.apply_action(state, action, pid)


def test_on_action_applied_sees_the_substituted_action_not_the_choice() -> None:
    """The hook's reason to be in the engine: an illegal choice is replaced, and the
    replacement is what was played — so it is what the hook must be told."""
    calls: list[tuple] = []
    engine = SimulationEngine(
        NimAdapter(n_sticks=11, max_take=3),
        [_AlwaysIllegalAgent(), RandomAgent(seed=0)],
        record=True,
        seed=0,
        on_action_applied=lambda *args: calls.append(args),
    )
    result = engine.run_game()
    assert result.actions is not None

    assert result.illegal_action_counts[0] > 0
    played_by_0 = [action for _, action, pid, _ in calls if pid == 0]
    assert played_by_0, "player 0 never played"
    assert all("__never_legal__" not in action.data for action in played_by_0)
    assert [a for _, a, _, _ in calls] == [a for _, _, a in result.actions]


def test_on_action_applied_is_not_called_when_apply_action_raises() -> None:
    calls: list[tuple] = []
    engine = SimulationEngine(
        _RaisingNimAdapter(n_sticks=11, max_take=3),
        [RandomAgent(seed=0), RandomAgent(seed=1)],
        on_action_applied=lambda *args: calls.append(args),
    )
    with pytest.raises(RuntimeError, match="refused"):
        engine.run_game()
    assert calls == []
