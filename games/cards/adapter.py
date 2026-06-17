from __future__ import annotations

import random

from core.base_adapter import BaseAdapter
from core.types import Action, ObservableState, State


class CardsAdapter(BaseAdapter):
    """A small 2-player trick-taking card game with hidden hands and chance.

    Setup: a 20-card deck (values 1..10, two copies each). Each player is dealt
    `hand_size` cards (default 5), kept hidden from the opponent. The remaining
    10 cards stay out of play but are unknown to both — they form an "unseen"
    pool that determinization must sample over (we know cards exist somewhere,
    just not where).

    Play: turns alternate in tricks of 2 cards. The trick leader plays first;
    the opponent responds knowing what was led. The higher card wins +1 point
    and leads the next trick. A tie discards both cards (0 points) and the
    previous leader leads again. Game ends when both hands are empty.

    State schema (the dict inside `State.data`):
        hands: {0: [...], 1: [...]} — full hands, full info
        scores: {0: int, 1: int}
        trick: list[int] — cards on the table in the in-progress trick (0 or 1)
        trick_players: list[int] — who played each card in `trick`
        leader: int — player who led the in-progress trick
        current: int — player to act next
        played: list[int] — all cards from completed (and discarded) tricks

    Designed as the canonical hidden-information adapter for the framework.
    """

    DECK_VALUES = tuple(range(1, 11))  # 1..10
    DECK_COPIES = 2

    def __init__(self, hand_size: int = 5, seed: int | None = None) -> None:
        if hand_size < 1:
            raise ValueError("hand_size must be >= 1")
        if hand_size * 2 > len(self.DECK_VALUES) * self.DECK_COPIES:
            raise ValueError("hand_size too large for the 20-card deck")
        self.hand_size = hand_size
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ lifecycle

    def init_state(self) -> State:
        """Deal a fresh game start. Engine reaches this via `get_initial_state`."""
        deck = [v for v in self.DECK_VALUES for _ in range(self.DECK_COPIES)]
        self._rng.shuffle(deck)
        return State(
            data={
                "hands": {
                    0: deck[: self.hand_size],
                    1: deck[self.hand_size : 2 * self.hand_size],
                },
                "scores": {0: 0, 1: 0},
                "trick": [],
                "trick_players": [],
                "leader": 0,
                "current": 0,
                "played": [],
            }
        )

    def get_initial_state(self) -> State:
        return self.init_state()

    def get_n_players(self) -> int:
        return 2

    # ------------------------------------------------------------------ queries

    def get_current_player(self, state: State) -> int:
        return int(state.data["current"])

    def is_terminal(self, state: State) -> bool:
        return all(len(h) == 0 for h in state.data["hands"].values())

    def get_legal_actions(self, state: State, player_id: int) -> list[Action]:
        hand = state.data["hands"][player_id]
        # Duplicates in the hand are strategically equivalent — collapse to uniques.
        return [Action(data={"card": v}) for v in sorted(set(hand))]

    def get_scores(self, state: State) -> dict[int, float]:
        return {pid: float(s) for pid, s in state.data["scores"].items()}

    # ------------------------------------------------------------------ transitions

    def apply_action(self, state: State, action: Action, player_id: int) -> State:
        new_state = self.clone_state(state)
        d = new_state.data
        card = int(action.data["card"])
        d["hands"][player_id].remove(card)
        d["trick"].append(card)
        d["trick_players"].append(player_id)

        if len(d["trick"]) == 2:
            c_first, c_second = d["trick"]
            p_first, p_second = d["trick_players"]
            if c_first > c_second:
                d["scores"][p_first] += 1
                d["leader"] = p_first
                d["current"] = p_first
            elif c_second > c_first:
                d["scores"][p_second] += 1
                d["leader"] = p_second
                d["current"] = p_second
            else:
                # Tie: cards discarded, same leader plays again.
                d["current"] = d["leader"]
            d["played"].extend(d["trick"])
            d["trick"] = []
            d["trick_players"] = []
        else:
            d["current"] = 1 - player_id

        return new_state

    # ------------------------------------------------------------------ visibility

    def get_observable_state(self, state: State, player_id: int) -> ObservableState:
        d = state.data
        opp_id = 1 - player_id
        obs_hands: dict[int, list] = {player_id: list(d["hands"][player_id]), opp_id: []}
        return ObservableState(
            data={
                "hands": obs_hands,
                "opp_hand_size": len(d["hands"][opp_id]),
                "scores": dict(d["scores"]),
                "trick": list(d["trick"]),
                "trick_players": list(d["trick_players"]),
                "leader": d["leader"],
                "current": d["current"],
                "played": list(d["played"]),
            },
            player_id=player_id,
        )

    def clone_state(self, state: State) -> State:
        d = state.data
        return State(
            data={
                "hands": {pid: list(hand) for pid, hand in d["hands"].items()},
                "scores": dict(d["scores"]),
                "trick": list(d["trick"]),
                "trick_players": list(d["trick_players"]),
                "leader": d["leader"],
                "current": d["current"],
                "played": list(d["played"]),
            }
        )

    def get_rich_renderable(self, obs_state: ObservableState):
        from rich.text import Text

        d = obs_state.data
        p = obs_state.player_id
        opp = 1 - p
        t = Text()
        t.append("Your hand: ", style="bold green")
        for card in sorted(d["hands"][p]):
            t.append(f"[{card}] ", style="bold cyan")
        t.append(f"  (opponent: {d['opp_hand_size']} card(s))\n", style="dim")
        if d["trick"]:
            t.append("\nTrick: ", style="dim")
            for card, pid in zip(d["trick"], d["trick_players"]):
                who = "you" if pid == p else f"P{pid}"
                t.append(f"{who}▸[{card}]  ", style="yellow" if pid == p else "red")
        scores = d["scores"]
        t.append("\n\nScore: ", style="dim")
        t.append(str(scores[p]), style="bold green")
        t.append(" — ", style="dim")
        t.append(str(scores[opp]), style="bold red")
        t.append(f"  (tricks played: {len(d['played'])})", style="dim")
        return t

    def get_action_display(self, action: Action) -> str:
        return f"Play [{action.data['card']}]"

    # ------------------------------------------------------------------ MCTS support

    def sample_state(self, observable_state: ObservableState) -> State:
        """Materialize a plausible full state consistent with `observable_state`.

        Wire this into `MCTSAgent(state_sampler=adapter.sample_state)` to enable
        per-simulation determinization. The opponent's hand is drawn uniformly
        from the unseen pool — i.e. the full deck minus our hand, minus every
        card already played, minus cards currently face-up in the in-progress
        trick. Cards left over after dealing the opponent represent the 10
        undealt cards and stay out of play.
        """
        d = observable_state.data
        my_id = observable_state.player_id
        opp_id = 1 - my_id

        unseen = [v for v in self.DECK_VALUES for _ in range(self.DECK_COPIES)]
        for c in list(d["hands"][my_id]) + list(d["played"]) + list(d["trick"]):
            unseen.remove(c)

        self._rng.shuffle(unseen)
        opp_hand = unseen[: d["opp_hand_size"]]

        hands: dict[int, list[int]] = {0: [], 1: []}
        hands[my_id] = list(d["hands"][my_id])
        hands[opp_id] = opp_hand

        return State(
            data={
                "hands": hands,
                "scores": dict(d["scores"]),
                "trick": list(d["trick"]),
                "trick_players": list(d["trick_players"]),
                "leader": d["leader"],
                "current": d["current"],
                "played": list(d["played"]),
            }
        )
