from __future__ import annotations

from core.base_adapter import BaseAdapter
from core.types import Action, ObservableState, State


class ConnectFourAdapter(BaseAdapter):
    """Connect Four, two players, configurable grid and win length.

    Discs fall to the lowest empty cell in the chosen column.
    First player to connect `connect` discs in a row (horizontally, vertically,
    or diagonally) wins. A full board with no winner is a draw.

    Args:
        rows:    number of rows (default 6)
        cols:    number of columns (default 7)
        connect: discs in a row needed to win (default 4)

    State schema:
        board         : list[int]   — rows*cols cells (row-major), 0=empty, 1=P0, 2=P1
        current_player: int         — player to move next
        game_over     : bool
        winner        : int | None  — 0, 1, or None (draw)
    """

    def __init__(self, rows: int = 6, cols: int = 7, connect: int = 4) -> None:
        if rows < connect and cols < connect:
            raise ValueError("Board too small for the required connect length")
        self.rows = rows
        self.cols = cols
        self.connect = connect

    def _idx(self, row: int, col: int) -> int:
        return row * self.cols + col

    def _check_win(self, board: list[int], last_row: int, last_col: int, token: int) -> bool:
        """Return True if `token` forms a connect-length run through (last_row, last_col)."""
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (1, -1):
                r, c = last_row + sign * dr, last_col + sign * dc
                while 0 <= r < self.rows and 0 <= c < self.cols and board[self._idx(r, c)] == token:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= self.connect:
                return True
        return False

    def get_initial_state(self) -> State:
        return State(
            data={
                "board": [0] * (self.rows * self.cols),
                "current_player": 0,
                "game_over": False,
                "winner": None,
            }
        )

    def get_n_players(self) -> int:
        return 2

    def get_current_player(self, state: State) -> int:
        return int(state.data["current_player"])

    def is_terminal(self, state: State) -> bool:
        return bool(state.data["game_over"])

    def get_legal_actions(self, state: State, player_id: int) -> list[Action]:
        board = state.data["board"]
        return [Action(data={"col": c}) for c in range(self.cols) if board[self._idx(0, c)] == 0]

    def apply_action(self, state: State, action: Action, player_id: int) -> State:
        col = int(action.data["col"])
        s = self.clone_state(state)
        board = s.data["board"]
        token = player_id + 1

        row = next(r for r in range(self.rows - 1, -1, -1) if board[self._idx(r, col)] == 0)
        board[self._idx(row, col)] = token

        if self._check_win(board, row, col, token):
            s.data["game_over"] = True
            s.data["winner"] = player_id
        elif all(board[self._idx(0, c)] != 0 for c in range(self.cols)):
            s.data["game_over"] = True
            # winner remains None → draw
        else:
            s.data["current_player"] = 1 - player_id

        return s

    def get_scores(self, state: State) -> dict[int, float]:
        winner = state.data["winner"]
        if winner is None:
            return {0: 0.5, 1: 0.5}
        return {winner: 1.0, 1 - winner: 0.0}

    def get_observable_state(self, state: State, player_id: int) -> ObservableState:
        d = state.data
        return ObservableState(
            data={
                "board": list(d["board"]),
                "current_player": d["current_player"],
                "game_over": d["game_over"],
                "winner": d["winner"],
            },
            player_id=player_id,
        )

    def clone_state(self, state: State) -> State:
        d = state.data
        return State(
            data={
                "board": list(d["board"]),
                "current_player": d["current_player"],
                "game_over": d["game_over"],
                "winner": d["winner"],
            }
        )

    def get_action_label(self, action: Action) -> str:
        return str(action.data["col"])
