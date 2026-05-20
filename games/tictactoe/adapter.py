from __future__ import annotations

import random
from typing import Literal

from core.base_adapter import BaseAdapter
from core.types import Action, ObservableState, State


_WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
]


def _winner(board: list[int]) -> int | None:
    """Return the winning player id (0 or 1), or None if no winner yet."""
    for a, b, c in _WIN_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a] - 1  # stored as 1/2, return 0/1
    return None


class TicTacToeAdapter(BaseAdapter):
    """Multi-round Tic Tac Toe on a 3×3 grid.

    Each round is an independent game; the session ends after `max_rounds`.
    The `starting_player` parameter controls who opens each round:
        "random"    — drawn uniformly at the start of each round.
        "alternate" — player 0 starts round 0, player 1 starts round 1, etc.
        "constant"  — player 0 always starts.

    Board cells are indexed 0–8 (row-major). Cell values: 0 = empty,
    1 = player 0's mark, 2 = player 1's mark.

    State schema:
        board         : list[int]        — 9 cells
        current       : int              — player to move
        round         : int              — current round index (0-based)
        round_starter : int              — player who opened the current round
        scores        : {0: int, 1: int} — cumulative round wins
        game_over     : bool
    """

    def __init__(
        self,
        starting_player: Literal["random", "alternate", "constant"] = "random",
        max_rounds: int = 9,
        seed: int | None = None,
    ) -> None:
        if starting_player not in ("random", "alternate", "constant"):
            raise ValueError("starting_player must be 'random', 'alternate', or 'constant'")
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self.starting_player = starting_player
        self.max_rounds = max_rounds
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ lifecycle

    def get_initial_state(self) -> State:
        starter = self._pick_starter(prev_starter=0, is_first=True)
        return State(
            data={
                "board": [0] * 9,
                "current": starter,
                "round": 0,
                "round_starter": starter,
                "scores": {0: 0, 1: 0},
                "game_over": False,
            }
        )

    def get_n_players(self) -> int:
        return 2

    # ------------------------------------------------------------------ queries

    def get_current_player(self, state: State) -> int:
        return int(state.data["current"])

    def is_terminal(self, state: State) -> bool:
        return bool(state.data["game_over"])

    def get_legal_actions(self, state: State, player_id: int) -> list[Action]:
        return [
            Action(data={"pos": i})
            for i, cell in enumerate(state.data["board"])
            if cell == 0
        ]

    def get_scores(self, state: State) -> dict[int, float]:
        return {pid: float(s) for pid, s in state.data["scores"].items()}

    # ------------------------------------------------------------------ transitions

    def apply_action(self, state: State, action: Action, player_id: int) -> State:
        s = self.clone_state(state)
        d = s.data
        pos = int(action.data["pos"])
        d["board"][pos] = player_id + 1  # 1 or 2

        w = _winner(d["board"])
        if w is not None:
            d["scores"][w] += 1
            self._end_round(d)
        elif all(cell != 0 for cell in d["board"]):
            self._end_round(d)
        else:
            d["current"] = 1 - player_id

        return s

    def _end_round(self, d: dict) -> None:
        d["round"] += 1
        if d["round"] >= self.max_rounds:
            d["game_over"] = True
            return
        next_starter = self._pick_starter(prev_starter=d["round_starter"], is_first=False)
        d["board"] = [0] * 9
        d["round_starter"] = next_starter
        d["current"] = next_starter

    def _pick_starter(self, prev_starter: int, is_first: bool) -> int:
        if self.starting_player == "random":
            return self._rng.randint(0, 1)
        if self.starting_player == "alternate":
            return 0 if is_first else 1 - prev_starter
        return 0  # constant

    # ------------------------------------------------------------------ visibility

    def get_observable_state(self, state: State, player_id: int) -> ObservableState:
        d = state.data
        return ObservableState(
            data={
                "board": list(d["board"]),
                "current": d["current"],
                "round": d["round"],
                "round_starter": d["round_starter"],
                "scores": dict(d["scores"]),
                "game_over": d["game_over"],
            },
            player_id=player_id,
        )

    def clone_state(self, state: State) -> State:
        d = state.data
        return State(
            data={
                "board": list(d["board"]),
                "current": d["current"],
                "round": d["round"],
                "round_starter": d["round_starter"],
                "scores": {pid: s for pid, s in d["scores"].items()},
                "game_over": d["game_over"],
            }
        )
