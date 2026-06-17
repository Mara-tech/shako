from __future__ import annotations

import pytest

from core.engine import SimulationEngine
from core.types import Action, State
from games.ludo.adapter import LudoAdapter, _STABLE
from rl.random_agent import RandomAgent

# Constants for the default track_size=52 board
_DONE = 57   # track_size + HOME_COL - 1 = 52 + 6 - 1
_LAST = 51   # last main-track square = track_size - 1


# ------------------------------------------------------------------ helpers

def _make_state(
    positions: list[list[int]],
    current: int = 0,
    dice: int = 1,
    game_over: bool = False,
    winner: int | None = None,
) -> State:
    return State(data={
        "positions": [list(row) for row in positions],
        "current": current,
        "dice": dice,
        "game_over": game_over,
        "winner": winner,
    })


def _blank(n: int = 4) -> list[list[int]]:
    return [[_STABLE] * 4 for _ in range(n)]


# ------------------------------------------------------------------ initial state

def test_initial_state_all_in_stable() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    s = adapter.get_initial_state()
    for player in range(4):
        assert all(p == _STABLE for p in s.data["positions"][player])


def test_initial_dice_valid() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    s = adapter.get_initial_state()
    assert 1 <= s.data["dice"] <= 6


# ------------------------------------------------------------------ constructor validation

def test_invalid_n_players() -> None:
    with pytest.raises(ValueError):
        LudoAdapter(n_players=1)
    with pytest.raises(ValueError):
        LudoAdapter(n_players=5)


def test_invalid_track_size() -> None:
    with pytest.raises(ValueError):
        LudoAdapter(track_size=50)  # not a multiple of 4
    with pytest.raises(ValueError):
        LudoAdapter(track_size=0)


def test_track_size_56_entry_points() -> None:
    adapter = LudoAdapter(n_players=4, track_size=56, seed=0)
    assert adapter._entries == [0, 14, 28, 42]
    assert adapter._done == 61  # 56 + 6 - 1


# ------------------------------------------------------------------ legal actions

def test_no_legal_move_all_stable_non_six() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    s = _make_state(_blank(), dice=3)
    actions = adapter.get_legal_actions(s, player_id=0)
    assert len(actions) == 1
    assert actions[0].data["piece"] == -1


def test_all_pieces_can_exit_on_six() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    s = _make_state(_blank(), dice=6)
    actions = adapter.get_legal_actions(s, player_id=0)
    pieces = sorted(a.data["piece"] for a in actions)
    assert pieces == [0, 1, 2, 3]


@pytest.mark.parametrize("dice", range(1, 6))
def test_no_exit_on_non_six(dice: int) -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    s = _make_state(_blank(), dice=dice)
    actions = adapter.get_legal_actions(s, player_id=0)
    assert all(a.data["piece"] == -1 for a in actions)


def test_no_overshoot_home_column() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    pos = _blank()
    pos[0][0] = 56  # 1 step from done
    s = _make_state(pos, dice=2)
    actions = adapter.get_legal_actions(s, player_id=0)
    pieces = [a.data["piece"] for a in actions]
    assert 0 not in pieces


def test_can_land_exactly_on_done() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    pos = _blank()
    pos[0][0] = 56
    s = _make_state(pos, dice=1)
    actions = adapter.get_legal_actions(s, player_id=0)
    pieces = [a.data["piece"] for a in actions]
    assert 0 in pieces


def test_finished_pieces_not_offered() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    pos = _blank()
    pos[0] = [_DONE, _DONE, _DONE, 5]
    s = _make_state(pos, dice=3)
    actions = adapter.get_legal_actions(s, player_id=0)
    pieces = [a.data["piece"] for a in actions]
    assert pieces == [3]


# ------------------------------------------------------------------ home column entry rule

def test_enter_home_column_from_last_square() -> None:
    # From position 51 (last main-track square), rolling 3 → home-column square 3
    adapter = LudoAdapter(n_players=4, seed=0)
    pos = _blank()
    pos[0][0] = _LAST
    s = _make_state(pos, dice=3)
    new_s = adapter.apply_action(s, Action(data={"piece": 0}), player_id=0)
    assert new_s.data["positions"][0][0] == _LAST + 3  # = 54


def test_cannot_enter_home_column_from_earlier_square() -> None:
    # From position 50, dice 4 would overshoot into home column — illegal
    adapter = LudoAdapter(n_players=4, seed=0)
    pos = _blank()
    pos[0][0] = 50
    s = _make_state(pos, dice=4)
    actions = adapter.get_legal_actions(s, player_id=0)
    pieces = [a.data["piece"] for a in actions]
    assert 0 not in pieces


def test_can_reach_last_main_track_square_then_stop() -> None:
    # From position 49, dice 2 → position 51 (still on main track)
    adapter = LudoAdapter(n_players=4, seed=0)
    pos = _blank()
    pos[0][0] = 49
    s = _make_state(pos, dice=2)
    actions = adapter.get_legal_actions(s, player_id=0)
    pieces = [a.data["piece"] for a in actions]
    assert 0 in pieces
    new_s = adapter.apply_action(s, Action(data={"piece": 0}), player_id=0)
    assert new_s.data["positions"][0][0] == _LAST


def test_home_column_entry_rule_with_track_size_56() -> None:
    adapter = LudoAdapter(n_players=4, track_size=56, seed=0)
    last = 55  # track_size - 1
    # done = 61  # track_size + HOME_COL - 1
    pos = _blank()
    pos[0][0] = last
    s = _make_state(pos, dice=3)
    new_s = adapter.apply_action(s, Action(data={"piece": 0}), player_id=0)
    assert new_s.data["positions"][0][0] == last + 3  # = 58

    pos2 = _blank()
    pos2[0][0] = 54  # one step before last
    s2 = _make_state(pos2, dice=3)
    actions = adapter.get_legal_actions(s2, player_id=0)
    assert all(a.data["piece"] == -1 for a in actions)  # can't skip into home col


# ------------------------------------------------------------------ immutability

def test_apply_action_does_not_mutate_state() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    pos = _blank()
    pos[0][0] = 5
    s = _make_state(pos, dice=3)
    adapter.apply_action(s, Action(data={"piece": 0}), player_id=0)
    assert s.data["positions"][0][0] == 5


# ------------------------------------------------------------------ movement

def test_exit_from_stable() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    s = _make_state(_blank(), dice=6)
    new_s = adapter.apply_action(s, Action(data={"piece": 0}), player_id=0)
    assert new_s.data["positions"][0][0] == 0


def test_advance_on_main_track() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    pos = _blank()
    pos[0][0] = 5
    s = _make_state(pos, dice=3)
    new_s = adapter.apply_action(s, Action(data={"piece": 0}), player_id=0)
    assert new_s.data["positions"][0][0] == 8


# ------------------------------------------------------------------ capture

def test_capture_sends_opponent_to_stable() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    pos = _blank()
    # Player 0 at rel 5 (abs 5) with dice 3 → rel 8 (abs 8)
    # Player 1 at abs 8 → rel for player 1: (8-13+52)%52 = 47
    pos[0][0] = 5
    pos[1][0] = 47
    s = _make_state(pos, dice=3)
    new_s = adapter.apply_action(s, Action(data={"piece": 0}), player_id=0)
    assert new_s.data["positions"][0][0] == 8
    assert new_s.data["positions"][1][0] == _STABLE


def test_no_capture_on_own_entry_square() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    pos = _blank()
    # Player 0 at rel 10 (abs 10) with dice 3 → rel 13 (abs 13)
    # Player 1 at rel 0 → abs 13 = player 1's own entry square — safe
    pos[0][0] = 10
    pos[1][0] = 0
    s = _make_state(pos, dice=3)
    new_s = adapter.apply_action(s, Action(data={"piece": 0}), player_id=0)
    assert new_s.data["positions"][0][0] == 13
    assert new_s.data["positions"][1][0] == 0


def test_home_column_safe_from_capture() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    pos = _blank()
    pos[0][0] = 20
    pos[1][0] = 53  # player 1 in home column — no abs position on main track
    s = _make_state(pos, dice=3)
    new_s = adapter.apply_action(s, Action(data={"piece": 0}), player_id=0)
    assert new_s.data["positions"][1][0] == 53


# ------------------------------------------------------------------ turn management

def test_six_keeps_same_player() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    pos = _blank()
    pos[0][0] = 5
    s = _make_state(pos, current=0, dice=6)
    new_s = adapter.apply_action(s, Action(data={"piece": 0}), player_id=0)
    assert new_s.data["current"] == 0


def test_non_six_advances_to_next_player() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    pos = _blank()
    pos[0][0] = 5
    s = _make_state(pos, current=0, dice=3)
    new_s = adapter.apply_action(s, Action(data={"piece": 0}), player_id=0)
    assert new_s.data["current"] == 1


def test_pass_advances_to_next_player() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    s = _make_state(_blank(), current=0, dice=3)
    new_s = adapter.apply_action(s, Action(data={"piece": -1}), player_id=0)
    assert new_s.data["current"] == 1


def test_forced_pass_on_six_does_not_grant_bonus() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    pos = _blank()
    pos[0] = [56, 56, 56, 56]  # dice 6 overshoots all pieces
    s = _make_state(pos, current=0, dice=6)
    new_s = adapter.apply_action(s, Action(data={"piece": -1}), player_id=0)
    assert new_s.data["current"] == 1


def test_turn_wraps_to_player_0() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    pos = _blank()
    pos[3][0] = 5
    s = _make_state(pos, current=3, dice=3)
    new_s = adapter.apply_action(s, Action(data={"piece": 0}), player_id=3)
    assert new_s.data["current"] == 0


# ------------------------------------------------------------------ win condition

def test_win_when_last_piece_reaches_done() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    pos = _blank()
    pos[0] = [_DONE, _DONE, _DONE, 56]
    s = _make_state(pos, current=0, dice=1)
    new_s = adapter.apply_action(s, Action(data={"piece": 3}), player_id=0)
    assert new_s.data["game_over"] is True
    assert new_s.data["winner"] == 0


def test_scores_at_terminal() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    s = _make_state(_blank(), game_over=True, winner=2)
    scores = adapter.get_scores(s)
    assert scores[2] == 1.0
    assert all(scores[p] == 0.0 for p in range(4) if p != 2)


def test_partial_scores_fraction_done() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    pos = _blank()
    pos[0] = [_DONE, _DONE, 5, _STABLE]  # 2 of 4 done
    s = _make_state(pos)
    scores = adapter.get_scores(s)
    assert scores[0] == pytest.approx(0.5)
    assert scores[1] == pytest.approx(0.0)


# ------------------------------------------------------------------ clone

def test_clone_is_independent() -> None:
    adapter = LudoAdapter(n_players=4, seed=0)
    s = adapter.get_initial_state()
    clone = adapter.clone_state(s)
    clone.data["positions"][0][0] = 10
    assert s.data["positions"][0][0] == _STABLE


# ------------------------------------------------------------------ integration

def test_random_4player_game_terminates() -> None:
    adapter = LudoAdapter(n_players=4, seed=42)
    agents = [RandomAgent(seed=i) for i in range(4)]
    engine = SimulationEngine(adapter, agents)
    result = engine.run_game()
    assert result.winner_id is not None
    assert result.n_turns > 0


def test_random_2player_game_terminates() -> None:
    adapter = LudoAdapter(n_players=2, seed=42)
    agents = [RandomAgent(seed=i) for i in range(2)]
    engine = SimulationEngine(adapter, agents)
    result = engine.run_game()
    assert result.winner_id is not None
    assert result.n_turns > 0


def test_random_game_track_size_56_terminates() -> None:
    adapter = LudoAdapter(n_players=4, track_size=56, seed=42)
    agents = [RandomAgent(seed=i) for i in range(4)]
    engine = SimulationEngine(adapter, agents)
    result = engine.run_game()
    assert result.winner_id is not None
