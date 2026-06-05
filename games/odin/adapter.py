from __future__ import annotations

import random
from itertools import combinations
from typing import Any

from core.base_adapter import BaseAdapter
from core.types import Action, ObservableState, State


def _value(cards: list[tuple[int, int]]) -> int:
    """Highest number formed by arranging card digits descending (e.g. 2+8 → 82)."""
    return int("".join(str(d) for d in sorted((c[0] for c in cards), reverse=True)))


def _homogeneous(cards: list[tuple[int, int]]) -> bool:
    """True iff all cards share the same number or all share the same colour."""
    if len(cards) <= 1:
        return True
    return len({c[0] for c in cards}) == 1 or len({c[1] for c in cards}) == 1


class OdinAdapter(BaseAdapter):
    """Odin: a 2–6 player shedding card game with coloured number cards.

    Components: 54 cards — numbers 1–9 in 6 colours (4 colours in the 2-player
    variant). Each player starts with 9 cards; remaining cards are set aside.

    Rounds: the round-starter plays exactly 1 card face-up (or all cards if
    their whole hand is homogeneous, ending the hand immediately). Each
    subsequent player either passes or plays a set of N or N+1 cards whose
    combined value strictly exceeds the current table — then picks up 1 card
    from the previous set. A multi-card set must share a number or a colour.
    Combined value: sort the card digits descending and read as one integer
    (e.g. 2 and 8 → 82; 2, 4, 9 → 942). If all but one player pass
    consecutively the round ends; the last player to place cards starts the
    next round. A hand ends the moment a player empties their hand (no pickup
    then) or a round-starter plays out a fully homogeneous hand. At hand's
    end every player scores 1 penalty point per card still held. First player
    to reach `points_limit` triggers game-over; lowest total penalty wins.

    Action payload:
        {"type": "play",  "cards": [(number, colour), ...], "pickup": (number, colour) | None}
            pickup is None at round-start (table was empty) or when the play
            empties the hand (rule: no pickup when hand becomes empty).
        {"type": "pass"}

    State schema:
        hands             : {pid: [(number, colour), ...]}
        scores            : {pid: int}   — cumulative penalty points
        table             : [(number, colour), ...]  — last played set
        round_starter     : int   — player who opens the current round
        last_placer       : int | None  — last player to place cards this round
        hand_starter      : int   — opener of the current hand (for rotation)
        current           : int   — player to act next
        consecutive_passes: int
        reserve           : [(number, colour), ...]
        n_players         : int
        game_over         : bool
        points_limit      : int

    Scores returned by `get_scores` are negated (higher = better) so the
    framework's maximisation convention applies to the penalty-based scoring.
    """

    def __init__(
        self,
        n_players: int = 4,
        points_limit: int = 15,
        seed: int | None = None,
    ) -> None:
        if not 2 <= n_players <= 6:
            raise ValueError("n_players must be between 2 and 6")
        self._n_players = n_players
        self._points_limit = points_limit
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ helpers

    def _deck(self) -> list[tuple[int, int]]:
        n_colours = 4 if self._n_players == 2 else 6
        return [(num, col) for col in range(n_colours) for num in range(1, 10)]

    def _deal(self, deck: list) -> tuple[dict[int, list], list]:
        hands = {pid: list(deck[pid * 9 : (pid + 1) * 9]) for pid in range(self._n_players)}
        return hands, list(deck[self._n_players * 9 :])

    # ------------------------------------------------------------------ lifecycle

    def get_initial_state(self) -> State:
        deck = self._deck()
        self._rng.shuffle(deck)
        hands, reserve = self._deal(deck)
        starter = self._rng.randint(0, self._n_players - 1)
        return State(
            data={
                "hands": hands,
                "scores": {pid: 0 for pid in range(self._n_players)},
                "table": [],
                "round_starter": starter,
                "last_placer": None,
                "hand_starter": starter,
                "current": starter,
                "consecutive_passes": 0,
                "reserve": reserve,
                "n_players": self._n_players,
                "game_over": False,
                "points_limit": self._points_limit,
            }
        )

    def get_n_players(self) -> int:
        return self._n_players

    # ------------------------------------------------------------------ queries

    def get_current_player(self, state: State) -> int:
        return int(state.data["current"])

    def is_terminal(self, state: State) -> bool:
        return bool(state.data["game_over"])

    def get_scores(self, state: State) -> dict[int, float]:
        return {pid: -float(s) for pid, s in state.data["scores"].items()}

    def get_legal_actions(self, state: State, player_id: int) -> list[Action]:
        d = state.data
        hand: list[tuple[int, int]] = d["hands"][player_id]
        table: list[tuple[int, int]] = d["table"]
        actions: list[Action] = []

        if not table:
            # Round start: play 1 card; or all cards if hand is homogeneous (optional).
            if _homogeneous(hand) and len(hand) > 1:
                actions.append(Action(data={"type": "play", "cards": list(hand), "pickup": None}))
            for card in {tuple(c) for c in hand}:
                actions.append(Action(data={"type": "play", "cards": [card], "pickup": None}))
        else:
            actions.append(Action(data={"type": "pass"}))
            n_table = len(table)
            tval = _value(table)
            seen: set[tuple] = set()
            for size in (n_table, n_table + 1):
                if size > len(hand):
                    continue
                for indices in combinations(range(len(hand)), size):
                    cards = [tuple(hand[i]) for i in indices]
                    key = tuple(sorted(cards))
                    if key in seen:
                        continue
                    seen.add(key)
                    if not _homogeneous(cards) or _value(cards) <= tval:
                        continue
                    idx_set = set(indices)
                    remaining = [hand[i] for i in range(len(hand)) if i not in idx_set]
                    if not remaining:
                        # Empties hand — no pickup allowed
                        actions.append(Action(data={"type": "play", "cards": cards, "pickup": None}))
                    else:
                        for pickup in {tuple(c) for c in table}:
                            actions.append(
                                Action(data={"type": "play", "cards": cards, "pickup": pickup})
                            )

        return actions

    # ------------------------------------------------------------------ transitions

    def apply_action(self, state: State, action: Action, player_id: int) -> State:
        s = self.clone_state(state)
        d = s.data
        n = d["n_players"]

        if action.data["type"] == "pass":
            d["consecutive_passes"] += 1
            if d["consecutive_passes"] >= n - 1:
                last = d["last_placer"] if d["last_placer"] is not None else d["round_starter"]
                d["table"] = []
                d["consecutive_passes"] = 0
                d["last_placer"] = None
                d["round_starter"] = last
                d["current"] = last
            else:
                d["current"] = (player_id + 1) % n
            return s

        # "play" action
        cards = [tuple(c) for c in action.data["cards"]]
        pickup = action.data.get("pickup")
        if pickup is not None:
            pickup = tuple(pickup)

        old_table = list(d["table"])
        hand = d["hands"][player_id]
        for card in cards:
            hand.remove(card)

        d["table"] = list(cards)
        d["last_placer"] = player_id
        d["consecutive_passes"] = 0

        if not hand:
            self._end_hand(d)
        else:
            if pickup is not None and old_table:
                hand.append(pickup)
            d["current"] = (player_id + 1) % n

        return s

    def _end_hand(self, d: dict) -> None:
        for pid, h in d["hands"].items():
            d["scores"][pid] += len(h)

        if any(s >= d["points_limit"] for s in d["scores"].values()):
            d["game_over"] = True
            return

        deck = self._deck()
        self._rng.shuffle(deck)
        new_starter = (d["hand_starter"] + 1) % d["n_players"]
        d["hand_starter"] = new_starter
        d["hands"], d["reserve"] = self._deal(deck)
        d["table"] = []
        d["round_starter"] = new_starter
        d["last_placer"] = None
        d["current"] = new_starter
        d["consecutive_passes"] = 0

    # ------------------------------------------------------------------ visibility

    def get_observable_state(self, state: State, player_id: int) -> ObservableState:
        d = state.data
        # Opponent hands are replaced by empty lists so that this dict can be
        # used as a valid State fallback by MCTS when no state_sampler is
        # provided (State(data=dict(obs.data)) must not KeyError on "hands").
        # Actual opponent sizes are kept separately for agents and sample_state.
        obs_hands: dict[int, list[Any]] = {pid: [] for pid in range(d["n_players"])}
        obs_hands[player_id] = list(d["hands"][player_id])
        return ObservableState(
            data={
                "hands": obs_hands,
                "hand_sizes": {
                    pid: len(d["hands"][pid])
                    for pid in range(d["n_players"])
                    if pid != player_id
                },
                "scores": dict(d["scores"]),
                "table": list(d["table"]),
                "round_starter": d["round_starter"],
                "last_placer": d["last_placer"],
                "hand_starter": d["hand_starter"],
                "current": d["current"],
                "consecutive_passes": d["consecutive_passes"],
                "reserve": [],
                "n_players": d["n_players"],
                "game_over": d["game_over"],
                "points_limit": d["points_limit"],
            },
            player_id=player_id,
        )

    def clone_state(self, state: State) -> State:
        d = state.data
        return State(
            data={
                "hands": {pid: list(h) for pid, h in d["hands"].items()},
                "scores": dict(d["scores"]),
                "table": list(d["table"]),
                "round_starter": d["round_starter"],
                "last_placer": d["last_placer"],
                "hand_starter": d["hand_starter"],
                "current": d["current"],
                "consecutive_passes": d["consecutive_passes"],
                "reserve": list(d["reserve"]),
                "n_players": d["n_players"],
                "game_over": d["game_over"],
                "points_limit": d["points_limit"],
            }
        )

    # ------------------------------------------------------------------ MCTS support

    def sample_state(self, observable_state: ObservableState) -> State:
        """Determinize: sample opponent hands uniformly from unseen cards.

        Wire into ``MCTSAgent(state_sampler=adapter.sample_state)`` for
        per-simulation determinization. Unseen = full deck minus own hand
        minus current table cards.
        """
        d = observable_state.data
        my_id = observable_state.player_id
        n = d["n_players"]

        full_deck = self._deck()
        my_hand = [tuple(c) for c in d["hands"][my_id]]
        seen = my_hand + [tuple(c) for c in d["table"]]
        unseen = list(full_deck)
        for card in seen:
            unseen.remove(card)
        self._rng.shuffle(unseen)

        hands: dict[int, list] = {my_id: my_hand}
        offset = 0
        for pid in range(n):
            if pid == my_id:
                continue
            size = d["hand_sizes"][pid]
            hands[pid] = list(unseen[offset : offset + size])
            offset += size

        return State(
            data={
                "hands": hands,
                "scores": dict(d["scores"]),
                "table": [tuple(c) for c in d["table"]],
                "round_starter": d["round_starter"],
                "last_placer": d.get("last_placer"),
                "hand_starter": d.get("hand_starter", 0),
                "current": d["current"],
                "consecutive_passes": d["consecutive_passes"],
                "reserve": list(unseen[offset:]),
                "n_players": n,
                "game_over": d["game_over"],
                "points_limit": d.get("points_limit", self._points_limit),
            }
        )
