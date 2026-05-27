from __future__ import annotations

import random

from core.base_adapter import BaseAdapter
from core.types import Action, ObservableState, State


class SevenFamiliesAdapter(BaseAdapter):
    """7 Familles (Happy Families) card game for 2+ players.

    Players collect complete families by asking opponents for specific cards.
    A player must hold at least one card of a family to request another from it.
    On a hit the card is transferred and the asker goes again; on a miss the
    asker draws from the deck (if available) and the turn advances.  When a
    player holds all n_components cards of a family they immediately book it.
    The game ends when all families are booked; the winner has the most books.

    Cards are encoded as integers: family_id * n_components + component_id.

    State schema:
        hands     : {pid: list[int]} — each player's hand (hidden from opponents)
        deck      : list[int]        — draw pile (face-down)
        books     : {pid: list[int]} — booked family ids per player
        current   : int              — player to act next
        game_over : bool
    """

    def __init__(
        self,
        n_players: int = 4,
        n_families: int = 7,
        n_components: int = 6,
        seed: int | None = None,
    ) -> None:
        if n_players < 2:
            raise ValueError("n_players must be >= 2")
        if n_families < 1:
            raise ValueError("n_families must be >= 1")
        if n_components < 2:
            raise ValueError("n_components must be >= 2")
        self.n_players = n_players
        self.n_families = n_families
        self.n_components = n_components
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ lifecycle

    def get_initial_state(self) -> State:
        total = self.n_families * self.n_components
        deck = list(range(total))
        self._rng.shuffle(deck)

        hand_size = total // self.n_players
        hands: dict[int, list[int]] = {
            pid: deck[pid * hand_size : (pid + 1) * hand_size]
            for pid in range(self.n_players)
        }
        d: dict = {
            "hands": hands,
            "deck": deck[self.n_players * hand_size :],
            "books": {pid: [] for pid in range(self.n_players)},
            "current": 0,
            "game_over": False,
        }
        for pid in range(self.n_players):
            self._book_check(d, pid)
        return State(data=d)

    def get_n_players(self) -> int:
        return self.n_players

    # ------------------------------------------------------------------ queries

    def get_current_player(self, state: State) -> int:
        return int(state.data["current"])

    def is_terminal(self, state: State) -> bool:
        return bool(state.data["game_over"])

    def get_legal_actions(self, state: State, player_id: int) -> list[Action]:
        d = state.data
        hand = d["hands"][player_id]

        if not hand:
            return [Action(data={"pass": True})]

        families_held: dict[int, set[int]] = {}
        for card in hand:
            fid, cid = divmod(card, self.n_components)
            families_held.setdefault(fid, set()).add(cid)

        actions = []
        for fid, owned in families_held.items():
            for cid in range(self.n_components):
                if cid not in owned:
                    for target in range(self.n_players):
                        if target != player_id:
                            actions.append(Action(data={
                                "target": target,
                                "family": fid,
                                "component": cid,
                            }))

        return actions if actions else [Action(data={"pass": True})]

    def get_scores(self, state: State) -> dict[int, float]:
        return {pid: float(len(b)) for pid, b in state.data["books"].items()}

    # ------------------------------------------------------------------ transitions

    def apply_action(self, state: State, action: Action, player_id: int) -> State:
        s = self.clone_state(state)
        d = s.data

        if action.data.get("pass"):
            d["current"] = self._next_active(d, player_id)
            return s

        target = int(action.data["target"])
        family = int(action.data["family"])
        component = int(action.data["component"])
        card = family * self.n_components + component

        if card in d["hands"][target]:
            d["hands"][target].remove(card)
            d["hands"][player_id].append(card)
            self._book_check(d, player_id)
            if d["game_over"]:
                return s
            # go again unless booking emptied the hand
            if not d["hands"][player_id]:
                d["current"] = self._next_active(d, player_id)
        else:
            if d["deck"]:
                d["hands"][player_id].append(d["deck"].pop())
                self._book_check(d, player_id)
            if not d["game_over"]:
                d["current"] = self._next_active(d, player_id)

        return s

    def _book_check(self, d: dict, player_id: int) -> None:
        """Book complete families in player_id's hand, mutating d in place."""
        hand = d["hands"][player_id]
        families: dict[int, list[int]] = {}
        for card in hand:
            fid, _ = divmod(card, self.n_components)
            families.setdefault(fid, []).append(card)

        for fid, cards in families.items():
            if len(cards) == self.n_components:
                for c in cards:
                    hand.remove(c)
                d["books"][player_id].append(fid)

        if sum(len(b) for b in d["books"].values()) == self.n_families:
            d["game_over"] = True

    def _next_active(self, d: dict, from_player: int) -> int:
        """Return the next player (after from_player) who has cards in hand."""
        pid = (from_player + 1) % self.n_players
        for _ in range(self.n_players - 1):
            if d["hands"][pid]:
                return pid
            pid = (pid + 1) % self.n_players
        # All others have empty hands: from_player is the last one with cards.
        return from_player

    # ------------------------------------------------------------------ visibility

    def get_observable_state(self, state: State, player_id: int) -> ObservableState:
        d = state.data
        # Observable hands: own hand is real, opponents' hands are empty lists
        # (sizes are exposed separately).  Keeping the "hands" key lets MCTS
        # use this ObservableState as a State directly when no state_sampler is
        # configured — get_legal_actions/apply_action both expect "hands".
        obs_hands = {
            pid: list(d["hands"][pid]) if pid == player_id else []
            for pid in range(self.n_players)
        }
        return ObservableState(
            data={
                "hands": obs_hands,
                "hand_sizes": {
                    pid: len(d["hands"][pid])
                    for pid in range(self.n_players)
                    if pid != player_id
                },
                "deck": [],          # face-down; size exposed separately
                "deck_size": len(d["deck"]),
                "books": {pid: list(b) for pid, b in d["books"].items()},
                "current": d["current"],
                "game_over": d["game_over"],
            },
            player_id=player_id,
        )

    def clone_state(self, state: State) -> State:
        d = state.data
        return State(
            data={
                "hands": {pid: list(h) for pid, h in d["hands"].items()},
                "deck": list(d["deck"]),
                "books": {pid: list(b) for pid, b in d["books"].items()},
                "current": d["current"],
                "game_over": d["game_over"],
            }
        )

    # ------------------------------------------------------------------ MCTS support

    def sample_state(self, observable_state: ObservableState) -> State:
        """Materialize a plausible full state for MCTS determinization.

        Unseen cards (opponents' hands + deck) are shuffled and distributed to
        match the observed hand sizes; any remainder becomes the deck.
        """
        d = observable_state.data
        my_id = observable_state.player_id

        booked: set[int] = {
            fid * self.n_components + c
            for fids in d["books"].values()
            for fid in fids
            for c in range(self.n_components)
        }
        my_hand = d["hands"][my_id]
        unseen = [
            card for card in range(self.n_families * self.n_components)
            if card not in my_hand and card not in booked
        ]
        self._rng.shuffle(unseen)

        hands: dict[int, list[int]] = {my_id: list(my_hand)}
        pos = 0
        for pid in range(self.n_players):
            if pid == my_id:
                continue
            size = d["hand_sizes"][pid]
            hands[pid] = unseen[pos : pos + size]
            pos += size

        return State(
            data={
                "hands": hands,
                "deck": unseen[pos:],
                "books": {pid: list(b) for pid, b in d["books"].items()},
                "current": d["current"],
                "game_over": d["game_over"],
            }
        )

    def get_action_label(self, action: Action) -> str:
        if action.data.get("pass"):
            return "pass"
        return f"ask_f{action.data['family']}_c{action.data['component']}"
