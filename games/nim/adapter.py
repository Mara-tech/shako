from __future__ import annotations

from core.base_adapter import BaseAdapter
from core.types import Action, ObservableState, State


class NimAdapter(BaseAdapter):
    """Single-pile Nim, two players.

    Rules:
      * A pile starts with `n_sticks` sticks.
      * Players alternate; on each turn the current player removes 1..`max_take`
        sticks (capped at the remaining pile size).
      * If `last_takes_wins` is True, taking the final stick wins.
      * If `last_takes_wins` is False (default, misère variant), taking the
        final stick loses.

    The state is `{"sticks": int, "current_player": int}` — fully observable,
    no hidden information, which makes Nim the canonical sanity-check adapter
    for the whole engine + agents stack.
    """

    def __init__(
        self,
        n_sticks: int = 21,
        max_take: int = 3,
        last_takes_wins: bool = False,
    ) -> None:
        if n_sticks < 1:
            raise ValueError("n_sticks must be >= 1")
        if max_take < 1:
            raise ValueError("max_take must be >= 1")
        self.n_sticks = n_sticks
        self.max_take = max_take
        self.last_takes_wins = last_takes_wins

    def get_initial_state(self) -> State:
        return State(data={"sticks": self.n_sticks, "current_player": 0})

    def get_n_players(self) -> int:
        return 2

    def get_current_player(self, state: State) -> int:
        return int(state.data["current_player"])

    def is_terminal(self, state: State) -> bool:
        return int(state.data["sticks"]) == 0

    def get_legal_actions(self, state: State, player_id: int) -> list[Action]:
        sticks = int(state.data["sticks"])
        upper = min(self.max_take, sticks)
        return [Action(data={"take": k}) for k in range(1, upper + 1)]

    def apply_action(self, state: State, action: Action, player_id: int) -> State:
        take = int(action.data["take"])
        return State(
            data={
                "sticks": int(state.data["sticks"]) - take,
                "current_player": 1 - player_id,
            }
        )

    def get_scores(self, state: State) -> dict[int, float]:
        # `current_player` is whose turn it would be next; the previous player
        # took the last stick.
        last_taker = 1 - int(state.data["current_player"])
        other = int(state.data["current_player"])
        if self.last_takes_wins:
            return {last_taker: 1.0, other: 0.0}
        return {last_taker: 0.0, other: 1.0}

    def get_observable_state(self, state: State, player_id: int) -> ObservableState:
        return ObservableState(data=dict(state.data), player_id=player_id)

    def clone_state(self, state: State) -> State:
        # All values are ints, so a shallow dict copy is a deep copy.
        return State(data=dict(state.data))

    def get_rich_renderable(self, obs_state: ObservableState):
        from rich.text import Text

        sticks = int(obs_state.data["sticks"])
        t = Text()
        t.append(f"Pile: {sticks} stick{'s' if sticks != 1 else ''}\n\n")
        visual = min(sticks, 50)
        t.append("  " + "│" * visual, style="bold yellow")
        if sticks > 50:
            t.append(f" +{sticks - 50} more", style="dim")
        t.append(f"\n\nMax take: {self.max_take}", style="dim")
        return t

    def get_action_display(self, action: Action) -> str:
        n = action.data["take"]
        return f"Take {n} stick{'s' if n != 1 else ''}"
