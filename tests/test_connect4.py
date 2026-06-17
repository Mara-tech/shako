from __future__ import annotations

import pytest

from core.engine import SimulationEngine
from core.types import State
from games.connect4.adapter import ConnectFourAdapter
from rl.random_agent import RandomAgent


def _state(adapter: ConnectFourAdapter, board: list[int], current_player: int = 0, game_over: bool = False, winner=None) -> State:
    return State(data={"board": board, "current_player": current_player, "game_over": game_over, "winner": winner})


# -------- initial state --------------------------------------------------------


def test_initial_board_empty():
    adapter = ConnectFourAdapter()
    state = adapter.get_initial_state()
    assert state.data["board"] == [0] * (adapter.rows * adapter.cols)
    assert state.data["current_player"] == 0
    assert not adapter.is_terminal(state)


def test_initial_legal_actions_all_columns():
    adapter = ConnectFourAdapter()
    state = adapter.get_initial_state()
    cols = [a.data["col"] for a in adapter.get_legal_actions(state, 0)]
    assert cols == list(range(adapter.cols))


# -------- drop mechanics -------------------------------------------------------


def test_disc_falls_to_bottom_row():
    adapter = ConnectFourAdapter()
    state = adapter.get_initial_state()
    action = next(a for a in adapter.get_legal_actions(state, 0) if a.data["col"] == 3)
    new_state = adapter.apply_action(state, action, 0)
    assert new_state.data["board"][adapter._idx(adapter.rows - 1, 3)] == 1


def test_disc_stacks_above_previous():
    adapter = ConnectFourAdapter()
    state = adapter.get_initial_state()
    action = next(a for a in adapter.get_legal_actions(state, 0) if a.data["col"] == 0)
    s1 = adapter.apply_action(state, action, 0)
    action2 = next(a for a in adapter.get_legal_actions(s1, 1) if a.data["col"] == 0)
    s2 = adapter.apply_action(s1, action2, 1)
    assert s2.data["board"][adapter._idx(adapter.rows - 1, 0)] == 1
    assert s2.data["board"][adapter._idx(adapter.rows - 2, 0)] == 2


def test_full_column_not_legal():
    adapter = ConnectFourAdapter()
    board = [0] * (adapter.rows * adapter.cols)
    for r in range(adapter.rows):
        board[adapter._idx(r, 2)] = 1
    state = _state(adapter, board)
    cols = [a.data["col"] for a in adapter.get_legal_actions(state, 0)]
    assert 2 not in cols
    assert len(cols) == adapter.cols - 1


# -------- win detection --------------------------------------------------------


def test_win_horizontal():
    adapter = ConnectFourAdapter()
    board = [0] * (adapter.rows * adapter.cols)
    for c in range(adapter.connect - 1):
        board[adapter._idx(adapter.rows - 1, c)] = 1
    state = _state(adapter, board)
    action = next(a for a in adapter.get_legal_actions(state, 0) if a.data["col"] == adapter.connect - 1)
    new_state = adapter.apply_action(state, action, 0)
    assert new_state.data["game_over"]
    assert new_state.data["winner"] == 0


def test_win_vertical():
    adapter = ConnectFourAdapter()
    board = [0] * (adapter.rows * adapter.cols)
    for r in range(adapter.rows - adapter.connect + 1, adapter.rows):
        board[adapter._idx(r, 4)] = 2
    state = _state(adapter, board, current_player=1)
    action = next(a for a in adapter.get_legal_actions(state, 1) if a.data["col"] == 4)
    new_state = adapter.apply_action(state, action, 1)
    assert new_state.data["game_over"]
    assert new_state.data["winner"] == 1


def test_win_diagonal():
    adapter = ConnectFourAdapter()
    # Player 0: tokens at (rows-1,0), (rows-2,1), (rows-3,2); dropping col 3 lands on rows-4
    board = [0] * (adapter.rows * adapter.cols)
    n = adapter.connect
    for i in range(n - 1):
        board[adapter._idx(adapter.rows - 1 - i, i)] = 1
    # Fill rows below target cell so disc for col n-1 lands at row rows-n
    for r in range(adapter.rows - n + 1, adapter.rows):
        board[adapter._idx(r, n - 1)] = 2
    state = _state(adapter, board)
    action = next(a for a in adapter.get_legal_actions(state, 0) if a.data["col"] == n - 1)
    new_state = adapter.apply_action(state, action, 0)
    assert new_state.data["game_over"]
    assert new_state.data["winner"] == 0


# -------- draw ----------------------------------------------------------------


def test_draw_full_board_no_winner():
    adapter = ConnectFourAdapter()
    # Checkerboard pattern: alternating tokens prevent any 4-in-a-row
    board = [((r + c) % 2) + 1 for r in range(adapter.rows) for c in range(adapter.cols)]
    state = _state(adapter, board, game_over=True, winner=None)
    assert adapter.is_terminal(state)
    assert adapter.get_scores(state) == {0: 0.5, 1: 0.5}


# -------- scores --------------------------------------------------------------


def test_scores_winner():
    adapter = ConnectFourAdapter()
    n = adapter.rows * adapter.cols
    state = _state(adapter, [0] * n, game_over=True, winner=0)
    assert adapter.get_scores(state) == {0: 1.0, 1: 0.0}

    state2 = _state(adapter, [0] * n, game_over=True, winner=1)
    assert adapter.get_scores(state2) == {0: 0.0, 1: 1.0}


# -------- observable state ----------------------------------------------------


def test_observable_state_mirrors_full_state():
    adapter = ConnectFourAdapter()
    state = adapter.get_initial_state()
    obs = adapter.get_observable_state(state, 0)
    assert obs.data["board"] == state.data["board"]
    assert obs.data["current_player"] == state.data["current_player"]
    assert obs.player_id == 0


# -------- clone ---------------------------------------------------------------


def test_clone_is_independent():
    adapter = ConnectFourAdapter()
    state = adapter.get_initial_state()
    clone = adapter.clone_state(state)
    clone.data["board"][0] = 99
    assert state.data["board"][0] == 0


# -------- engine integration --------------------------------------------------


def test_random_game_terminates():
    adapter = ConnectFourAdapter()
    agents = [RandomAgent(seed=42), RandomAgent(seed=7)]
    engine = SimulationEngine(adapter, agents)
    result = engine.run_game()
    assert result.winner_id in (0, 1, None)
    assert result.n_turns <= adapter.rows * adapter.cols


def test_random_games_produce_both_winners():
    adapter = ConnectFourAdapter()
    winners: set[int | None] = set()
    for seed in range(50):
        agents = [RandomAgent(seed=seed), RandomAgent(seed=seed + 1000)]
        engine = SimulationEngine(adapter, agents)
        result = engine.run_game()
        winners.add(result.winner_id)
        if len(winners) >= 2:
            break
    assert len(winners) >= 2


# -------- non-standard configurations ----------------------------------------


@pytest.mark.parametrize("rows,cols,connect", [
    (5, 6, 3),
    (7, 8, 5),
])
def test_custom_config_game_terminates(rows: int, cols: int, connect: int):
    adapter = ConnectFourAdapter(rows=rows, cols=cols, connect=connect)
    agents = [RandomAgent(seed=0), RandomAgent(seed=1)]
    engine = SimulationEngine(adapter, agents)
    result = engine.run_game()
    assert result.winner_id in (0, 1, None)
    assert result.n_turns <= rows * cols
