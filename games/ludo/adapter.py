from __future__ import annotations

import random

from core.base_adapter import BaseAdapter
from core.types import Action, ObservableState, State

_PIECES = 4     # horses per player
_HOME_COL = 6   # squares in each player's private home column
_STABLE = -1    # piece not yet on the board


class LudoAdapter(BaseAdapter):
    """Ludo (Les Petits Chevaux) for 2–4 players.

    Each player has 4 horses starting in the stable. A roll of 6 is needed to
    exit. Horses travel track_size + 5 steps: track_size squares on the shared
    main track, then 6 in a private home column. First player to bring all 4
    horses home wins.

    Rolling a 6 grants a bonus turn, provided a real move was made.
    A horse standing on its own entry square is safe from capture.
    A horse can only enter the home column from the last main-track square
    (position track_size - 1); rolling i from there places the horse on
    home-column square i.

    The four entry squares are evenly spaced: track_size // 4 apart.
    Standard Ludo uses track_size=52 (entries at 0, 13, 26, 39).

    State schema:
        positions : list[list[int]]  — [player][piece]: -1=stable,
                                       0..track_size-1 = main track,
                                       track_size..track_size+5 = home column
                                       (track_size+5 = done)
        current   : int              — player to move
        dice      : int              — pre-rolled die (1–6) for current player
        game_over : bool
        winner    : int | None
    """

    def __init__(
        self,
        n_players: int = 4,
        track_size: int = 52,
        seed: int | None = None,
    ) -> None:
        if not 2 <= n_players <= 4:
            raise ValueError("n_players must be 2, 3, or 4")
        if track_size < 4 or track_size % 4 != 0:
            raise ValueError("track_size must be a positive multiple of 4")
        self.n_players = n_players
        self.track_size = track_size
        self._entries = [i * track_size // 4 for i in range(4)]
        self._done = track_size + _HOME_COL - 1
        self._rng = random.Random(seed)

    def _roll(self) -> int:
        return self._rng.randint(1, 6)

    def _abs_pos(self, player: int, rel: int) -> int:
        return (self._entries[player] + rel) % self.track_size

    # ------------------------------------------------------------------ lifecycle

    def get_initial_state(self) -> State:
        return State(data={
            "positions": [[_STABLE] * _PIECES for _ in range(self.n_players)],
            "current": 0,
            "dice": self._roll(),
            "game_over": False,
            "winner": None,
        })

    def get_n_players(self) -> int:
        return self.n_players

    # ------------------------------------------------------------------ queries

    def get_current_player(self, state: State) -> int:
        return int(state.data["current"])

    def is_terminal(self, state: State) -> bool:
        return bool(state.data["game_over"])

    def get_legal_actions(self, state: State, player_id: int) -> list[Action]:
        pos = state.data["positions"][player_id]
        dice = int(state.data["dice"])
        last_main = self.track_size - 1
        actions = []
        for i, p in enumerate(pos):
            if p == _STABLE:
                if dice == 6:
                    actions.append(Action(data={"piece": i}))
            elif p < self._done:
                new = p + dice
                # Home column entry only from the last main-track square.
                # Rolling i from last_main lands on home-column square i.
                if p < self.track_size and new >= self.track_size and p != last_main:
                    continue
                if new <= self._done:
                    actions.append(Action(data={"piece": i}))
        return actions or [Action(data={"piece": -1})]

    def get_scores(self, state: State) -> dict[int, float]:
        winner = state.data["winner"]
        if winner is not None:
            return {p: 1.0 if p == winner else 0.0 for p in range(self.n_players)}
        return {
            p: sum(1 for x in state.data["positions"][p] if x == self._done) / _PIECES
            for p in range(self.n_players)
        }

    # ------------------------------------------------------------------ transitions

    def apply_action(self, state: State, action: Action, player_id: int) -> State:
        s = self.clone_state(state)
        d = s.data
        dice = int(d["dice"])
        piece = int(action.data["piece"])

        if piece >= 0:
            pos = d["positions"][player_id]
            old = pos[piece]
            new = 0 if old == _STABLE else old + dice
            pos[piece] = new

            # Captures: only on main track; a piece on its own entry square is safe
            if new < self.track_size:
                abs_new = self._abs_pos(player_id, new)
                for opp in range(self.n_players):
                    if opp == player_id:
                        continue
                    for j, op in enumerate(d["positions"][opp]):
                        if (
                            op != _STABLE
                            and op < self.track_size
                            and self._abs_pos(opp, op) == abs_new
                            and abs_new != self._entries[opp]
                        ):
                            d["positions"][opp][j] = _STABLE

            # Win check
            if all(x == self._done for x in d["positions"][player_id]):
                d["game_over"] = True
                d["winner"] = player_id
                return s

        # Same player rolls again on 6 (only when a real move was made)
        if dice == 6 and piece >= 0:
            d["dice"] = self._roll()
        else:
            d["current"] = (player_id + 1) % self.n_players
            d["dice"] = self._roll()

        return s

    # ------------------------------------------------------------------ visibility

    def get_observable_state(self, state: State, player_id: int) -> ObservableState:
        d = state.data
        return ObservableState(
            data={
                "positions": [list(row) for row in d["positions"]],
                "current": d["current"],
                "dice": d["dice"],
                "game_over": d["game_over"],
                "winner": d["winner"],
            },
            player_id=player_id,
        )

    def clone_state(self, state: State) -> State:
        d = state.data
        return State(data={
            "positions": [list(row) for row in d["positions"]],
            "current": d["current"],
            "dice": d["dice"],
            "game_over": d["game_over"],
            "winner": d["winner"],
        })

    def get_action_label(self, action: Action) -> str:
        return "pass" if action.data["piece"] < 0 else "move"
