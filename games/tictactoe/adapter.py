from __future__ import annotations

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
    """Single-episode Tic Tac Toe on a 3×3 grid.

    Playing several games in a row (a "match"), tallying the outcomes, and
    detecting seat advantage are concerns of the caller (the CLI's replay
    loop, `SelfPlayTrainer`, `DominanceAnalyzer`, ...), not of the adapter —
    every one of them already aggregates independent `GameResult`s. Baking a
    multi-round session into the adapter's own state would instead dilute the
    per-move reward signal MCTS relies on (a move's value would depend on many
    future, unrelated rounds) and would hide seat-advantage from analysis
    tools that only see one aggregated result per session.

    Player 0 always moves first, same convention as every other adapter in the
    framework (`nim`, `cards`, `connect4`). Deciding who effectively "goes
    first" against a given opponent is a seat-assignment concern of the
    caller (e.g. the CLI's seat prompt, `SelfPlayTrainer`'s alternating
    seats in `_evaluate`) — not something the adapter needs to parameterize.

    Board cells are indexed 0–8 (row-major). Cell values: 0 = empty,
    1 = player 0's mark, 2 = player 1's mark.

    State schema:
        board     : list[int]            — 9 cells
        current   : int                  — player to move
        scores    : {0: float, 1: float} — this game's outcome
                     (win = 1.0 / 0.0, draw = 0.5 / 0.5, in progress = 0.0 / 0.0)
        game_over : bool
    """

    def get_initial_state(self) -> State:
        return State(
            data={
                "board": [0] * 9,
                "current": 0,
                "scores": {0: 0.0, 1: 0.0},
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
            d["scores"][w] = 1.0
            d["game_over"] = True
        elif all(cell != 0 for cell in d["board"]):
            # Match-point scoring: a draw is worth half a win to each player,
            # so MCTS backprop can tell "secured a draw" apart from "lost"
            # (both would otherwise leave the mover's own score at 0).
            d["scores"][0] = 0.5
            d["scores"][1] = 0.5
            d["game_over"] = True
        else:
            d["current"] = 1 - player_id

        return s

    # ------------------------------------------------------------------ visibility

    def get_observable_state(self, state: State, player_id: int) -> ObservableState:
        d = state.data
        return ObservableState(
            data={
                "board": list(d["board"]),
                "current": d["current"],
                "scores": dict(d["scores"]),
                "game_over": d["game_over"],
            },
            player_id=player_id,
        )

    def get_rich_renderable(self, obs_state: ObservableState):
        from rich.text import Text

        board = obs_state.data["board"]
        syms = ["·", "X", "O"]
        cols = ["dim", "bold red", "bold blue"]
        t = Text()
        for r in range(3):
            for c in range(3):
                if c:
                    t.append(" │ ", style="dim")
                val = board[r * 3 + c]
                t.append(syms[val], style=cols[val])
            if r < 2:
                t.append("\n───┼───┼───\n", style="dim")
        return t

    def get_grid_config(self) -> dict:
        return {"rows": 3, "cols": 3, "mode": "cell"}

    def get_grid_render_config(self) -> dict:
        return {
            "symbols": {0: "·", 1: "X", 2: "O"},
            "colors": {0: "dim", 1: "bold red", 2: "bold blue"},
        }

    def get_grid_info(self, obs_state: ObservableState) -> dict:
        return {"board": obs_state.data["board"]}

    def get_action_for_click(
        self, row: int, col: int, legal_actions: list[Action]
    ) -> Action | None:
        pos = row * 3 + col
        for a in legal_actions:
            if a.data["pos"] == pos:
                return a
        return None

    def get_action_display(self, action: Action) -> str:
        pos = action.data["pos"]
        r, c = divmod(pos, 3)
        names = ["Top-left", "Top-center", "Top-right",
                 "Mid-left", "Center", "Mid-right",
                 "Bot-left", "Bot-center", "Bot-right"]
        return f"{names[pos]} ({r},{c})"

    def clone_state(self, state: State) -> State:
        d = state.data
        return State(
            data={
                "board": list(d["board"]),
                "current": d["current"],
                "scores": dict(d["scores"]),
                "game_over": d["game_over"],
            }
        )
