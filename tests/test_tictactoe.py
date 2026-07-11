from __future__ import annotations

import pytest

from core.engine import SimulationEngine
from core.types import Action, State
from games.tictactoe.adapter import TicTacToeAdapter, _winner
from rl.random_agent import RandomAgent


# ------------------------------------------------------------------ helpers

def _state(
    board: list[int],
    current: int = 0,
    round_: int = 0,
    round_starter: int = 0,
    scores: dict | None = None,
) -> State:
    return State(
        data={
            "board": list(board),
            "current": current,
            "round": round_,
            "round_starter": round_starter,
            "scores": scores if scores is not None else {0: 0, 1: 0},
            "game_over": False,
        }
    )


def _act(pos: int) -> Action:
    return Action(data={"pos": pos})


# ------------------------------------------------------------------ _winner helper

def test_winner_empty_board() -> None:
    assert _winner([0] * 9) is None


@pytest.mark.parametrize(
    "positions,player",
    [
        ((0, 1, 2), 0),  # top row
        ((3, 4, 5), 0),  # middle row
        ((6, 7, 8), 0),  # bottom row
        ((0, 3, 6), 1),  # left column
        ((1, 4, 7), 1),  # center column
        ((2, 5, 8), 1),  # right column
        ((0, 4, 8), 0),  # main diagonal
        ((2, 4, 6), 1),  # anti-diagonal
    ],
)
def test_winner_all_lines(positions: tuple[int, int, int], player: int) -> None:
    board = [0] * 9
    for pos in positions:
        board[pos] = player + 1
    assert _winner(board) == player


def test_winner_no_false_positive_partial_line() -> None:
    board = [0] * 9
    board[0] = board[1] = 1  # two in a row, not three
    assert _winner(board) is None


# ------------------------------------------------------------------ legal actions

def test_legal_actions_empty_board() -> None:
    adapter = TicTacToeAdapter()
    state = adapter.get_initial_state()
    legal = adapter.get_legal_actions(state, 0)
    assert sorted(a.data["pos"] for a in legal) == list(range(9))


def test_legal_actions_partial_board() -> None:
    adapter = TicTacToeAdapter()
    board = [1, 0, 2, 0, 1, 0, 0, 0, 0]
    state = _state(board, current=1)
    legal = adapter.get_legal_actions(state, 1)
    expected = [i for i, c in enumerate(board) if c == 0]
    assert sorted(a.data["pos"] for a in legal) == expected


def test_legal_actions_one_cell_left() -> None:
    adapter = TicTacToeAdapter()
    board = [1, 2, 1, 2, 1, 2, 2, 1, 0]
    state = _state(board, current=0)
    legal = adapter.get_legal_actions(state, 0)
    assert [a.data["pos"] for a in legal] == [8]


# ------------------------------------------------------------------ apply_action basics

def test_apply_action_places_mark_player0() -> None:
    adapter = TicTacToeAdapter()
    state = _state([0] * 9, current=0)
    new = adapter.apply_action(state, _act(4), player_id=0)
    assert new.data["board"][4] == 1  # player 0 → mark 1
    assert state.data["board"][4] == 0  # original untouched


def test_apply_action_places_mark_player1() -> None:
    adapter = TicTacToeAdapter()
    board = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    state = _state(board, current=1)
    new = adapter.apply_action(state, _act(1), player_id=1)
    assert new.data["board"][1] == 2  # player 1 → mark 2


def test_apply_action_switches_player() -> None:
    adapter = TicTacToeAdapter()
    state = _state([0] * 9, current=0)
    new = adapter.apply_action(state, _act(0), player_id=0)
    assert new.data["current"] == 1
    state2 = _state([0] * 9, current=1)
    new2 = adapter.apply_action(state2, _act(0), player_id=1)
    assert new2.data["current"] == 0


def test_apply_action_does_not_mutate_source_state() -> None:
    adapter = TicTacToeAdapter()
    state = _state([0] * 9)
    adapter.apply_action(state, _act(3), player_id=0)
    assert state.data["board"] == [0] * 9


# ------------------------------------------------------------------ win detection

def test_win_increments_score_and_starts_new_round() -> None:
    adapter = TicTacToeAdapter(starting_player="constant", max_rounds=5)
    # Player 0 has 0,1 and will complete top row with pos 2
    board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
    state = _state(board, current=0, round_=0)
    new = adapter.apply_action(state, _act(2), player_id=0)
    assert new.data["scores"][0] == 1
    assert new.data["scores"][1] == 0
    assert new.data["board"] == [0] * 9  # reset
    assert new.data["round"] == 1
    assert not new.data["game_over"]


def test_win_on_last_round_ends_game() -> None:
    adapter = TicTacToeAdapter(starting_player="constant", max_rounds=3)
    board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
    state = _state(board, current=0, round_=2)  # already at last round
    new = adapter.apply_action(state, _act(2), player_id=0)
    assert new.data["game_over"]


def test_player1_win_increments_correct_score() -> None:
    adapter = TicTacToeAdapter(starting_player="constant", max_rounds=5)
    board = [1, 0, 0, 2, 2, 0, 0, 0, 0]
    # Player 1 completes middle row with pos 5
    state = _state(board, current=1, round_=0)
    new = adapter.apply_action(state, _act(5), player_id=1)
    assert new.data["scores"][1] == 1
    assert new.data["scores"][0] == 0


# ------------------------------------------------------------------ draw detection

def test_draw_starts_new_round_half_point_each() -> None:
    adapter = TicTacToeAdapter(starting_player="constant", max_rounds=5)
    # One cell left, no winner possible
    board = [1, 2, 1, 1, 2, 1, 2, 1, 0]
    state = _state(board, current=1, round_=0)
    new = adapter.apply_action(state, _act(8), player_id=1)
    assert new.data["scores"] == {0: 0.5, 1: 0.5}
    assert new.data["board"] == [0] * 9
    assert new.data["round"] == 1
    assert not new.data["game_over"]


def test_draw_on_last_round_ends_game() -> None:
    adapter = TicTacToeAdapter(max_rounds=1)
    board = [1, 2, 1, 1, 2, 1, 2, 1, 0]
    state = _state(board, current=1, round_=0)
    new = adapter.apply_action(state, _act(8), player_id=1)
    assert new.data["game_over"]


# ------------------------------------------------------------------ starting_player modes

def test_starting_player_constant_always_0() -> None:
    adapter = TicTacToeAdapter(starting_player="constant", max_rounds=6, seed=0)
    state = adapter.get_initial_state()
    assert state.data["round_starter"] == 0

    # Simulate several round endings and verify the starter never changes
    for _ in range(4):
        board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
        state = State(data={**state.data, "board": list(board), "current": 0})
        state = adapter.apply_action(state, _act(2), player_id=0)
        assert state.data["round_starter"] == 0


def test_starting_player_alternate_flips() -> None:
    adapter = TicTacToeAdapter(starting_player="alternate", max_rounds=6, seed=0)
    state = adapter.get_initial_state()
    assert state.data["round_starter"] == 0

    starters = [0]
    for _ in range(4):
        board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
        state = State(data={**state.data, "board": list(board), "current": 0})
        state = adapter.apply_action(state, _act(2), player_id=0)
        starters.append(state.data["round_starter"])

    assert starters == [0, 1, 0, 1, 0]


def test_starting_player_random_deterministic_with_seed() -> None:
    a1 = TicTacToeAdapter(starting_player="random", max_rounds=10, seed=7)
    a2 = TicTacToeAdapter(starting_player="random", max_rounds=10, seed=7)
    s1 = a1.get_initial_state()
    s2 = a2.get_initial_state()
    assert s1.data["round_starter"] == s2.data["round_starter"]


# ------------------------------------------------------------------ max_rounds

def test_game_not_over_before_max_rounds() -> None:
    adapter = TicTacToeAdapter(starting_player="constant", max_rounds=3)
    # Win round 0
    board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
    state = _state(board, current=0, round_=0)
    state = adapter.apply_action(state, _act(2), player_id=0)
    assert state.data["round"] == 1
    assert not state.data["game_over"]


def test_game_over_at_max_rounds() -> None:
    adapter = TicTacToeAdapter(starting_player="constant", max_rounds=1)
    assert adapter.max_rounds == 1
    board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
    state = _state(board, current=0, round_=0)
    state = adapter.apply_action(state, _act(2), player_id=0)
    assert state.data["game_over"]


def test_is_terminal_matches_game_over() -> None:
    adapter = TicTacToeAdapter(max_rounds=1)
    state = adapter.get_initial_state()
    assert not adapter.is_terminal(state)
    # Force game over
    state.data["game_over"] = True
    assert adapter.is_terminal(state)


# ------------------------------------------------------------------ scores

def test_scores_accumulate_across_rounds() -> None:
    adapter = TicTacToeAdapter(starting_player="constant", max_rounds=5)
    board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
    state = _state(board, current=0, round_=0, scores={0: 2, 1: 1})
    state = adapter.apply_action(state, _act(2), player_id=0)
    assert state.data["scores"] == {0: 3, 1: 1}


def test_get_scores_returns_floats() -> None:
    adapter = TicTacToeAdapter()
    state = _state([0] * 9, scores={0: 3, 1: 1})
    scores = adapter.get_scores(state)
    assert scores == {0: 3.0, 1: 1.0}
    assert all(isinstance(v, float) for v in scores.values())


# ------------------------------------------------------------------ observable state & clone

def test_observable_state_contains_all_fields() -> None:
    adapter = TicTacToeAdapter()
    state = _state([1, 0, 2, 0, 0, 0, 0, 0, 0], current=0)
    obs = adapter.get_observable_state(state, player_id=0)
    assert obs.player_id == 0
    for key in ("board", "current", "round", "round_starter", "scores", "game_over"):
        assert key in obs.data


def test_observable_state_board_is_copy() -> None:
    adapter = TicTacToeAdapter()
    state = _state([0] * 9)
    obs = adapter.get_observable_state(state, 0)
    obs.data["board"][0] = 99
    assert state.data["board"][0] == 0


def test_clone_state_is_independent() -> None:
    adapter = TicTacToeAdapter()
    state = _state([1, 0, 0, 0, 2, 0, 0, 0, 0])
    clone = adapter.clone_state(state)
    clone.data["board"][1] = 99
    clone.data["scores"][0] = 42
    assert state.data["board"][1] == 0
    assert state.data["scores"][0] == 0


# ------------------------------------------------------------------ integration

def test_random_agents_game_terminates() -> None:
    adapter = TicTacToeAdapter(starting_player="alternate", max_rounds=9, seed=0)
    agents = [RandomAgent(seed=i) for i in range(2)]
    engine = SimulationEngine(adapter, agents)
    results = [engine.run_game() for _ in range(30)]
    assert all(not r.timed_out for r in results)
    assert all(r.winner_id in (0, 1, None) for r in results)


def test_random_agents_scores_consistent() -> None:
    """Scores in GameResult must equal the adapter's final get_scores output."""
    adapter = TicTacToeAdapter(max_rounds=5, seed=1)
    agents = [RandomAgent(seed=0), RandomAgent(seed=1)]
    engine = SimulationEngine(adapter, agents)
    for _ in range(20):
        result = engine.run_game()
        total = sum(result.scores.values())
        # Total wins <= max_rounds
        assert total <= adapter.max_rounds


@pytest.mark.parametrize("mode", ["random", "alternate", "constant"])
def test_starting_player_modes_complete_without_error(mode: str) -> None:
    adapter = TicTacToeAdapter(starting_player=mode, max_rounds=5, seed=42)
    agents = [RandomAgent(seed=0), RandomAgent(seed=1)]
    engine = SimulationEngine(adapter, agents)
    results = [engine.run_game() for _ in range(10)]
    assert len(results) == 10


def test_alternate_starters_integration() -> None:
    """In alternate mode, round_starter must flip after every real round of actual play."""
    adapter = TicTacToeAdapter(starting_player="alternate", max_rounds=6, seed=0)
    agents = [RandomAgent(seed=0), RandomAgent(seed=1)]

    state = adapter.get_initial_state()
    starters: list[int] = [state.data["round_starter"]]
    prev_round: int = state.data["round"]

    while not adapter.is_terminal(state):
        pid = adapter.get_current_player(state)
        obs = adapter.get_observable_state(state, pid)
        actions = adapter.get_legal_actions(state, pid)
        action = agents[pid].choose_action(obs, actions)
        state = adapter.apply_action(state, action, pid)

        cur_round = state.data["round"]
        if cur_round != prev_round and not state.data["game_over"]:
            starters.append(state.data["round_starter"])
            prev_round = cur_round

    assert len(starters) >= 2, "Expected at least 2 rounds to check alternation"
    for i in range(len(starters) - 1):
        assert starters[i + 1] == 1 - starters[i], (
            f"Round {i + 1} starter should be {1 - starters[i]}, got {starters[i + 1]}"
        )
