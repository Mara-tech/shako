from __future__ import annotations

from typing import Callable

from core.base_agent import BaseAgent
from core.types import Action, ObservableState


RenderStateFn = Callable[[ObservableState], str]
RenderActionFn = Callable[[Action], str]
InputFn = Callable[[str], str]


class HumanAgent(BaseAgent):
    """Console-based agent driven by a human at the keyboard.

    Each turn it prints the observable state, lists every legal action with a
    numeric index, and reads an index from stdin. Invalid input (non-numeric,
    out of range) is rejected with a message and the prompt is repeated.

    Customize the display per game by passing `render_state_fn` and/or
    `render_action_fn`; the defaults just dump the underlying dicts.

    Do not combine with `SimulationEngine(max_action_ms=...)` — `input()`
    blocks until Enter is pressed and cannot be interrupted by the timeout.
    """

    def __init__(
        self,
        render_state_fn: RenderStateFn | None = None,
        render_action_fn: RenderActionFn | None = None,
        input_fn: InputFn = input,
    ) -> None:
        """Args:
            render_state_fn: optional formatter for `ObservableState`.
            render_action_fn: optional formatter for individual `Action`s.
            input_fn: prompt-and-read function (injectable for tests).
        """
        self._render_state = render_state_fn or self._default_render_state
        self._render_action = render_action_fn or self._default_render_action
        self._input = input_fn
        self.player_id: int = 0

    def on_game_start(self, player_id: int, n_players: int) -> None:
        self.player_id = player_id
        print(f"\n=== New game — you are player {player_id} of {n_players} ===")

    def on_game_end(self, scores: dict[int, float]) -> None:
        print(f"\n=== Game over — final scores: {scores} ===")

    def choose_action(
        self,
        observable_state: ObservableState,
        legal_actions: list[Action],
    ) -> Action:
        print()
        print(self._render_state(observable_state))
        print("\nLegal actions:")
        for i, action in enumerate(legal_actions):
            print(f"  [{i:>3}] {self._render_action(action)}")

        max_idx = len(legal_actions) - 1
        prompt = f"Your move [0-{max_idx}]: "
        while True:
            raw = self._input(prompt).strip()
            if not raw:
                print("Empty input — type a number.")
                continue
            try:
                idx = int(raw)
            except ValueError:
                print(f"'{raw}' is not a valid integer.")
                continue
            if not 0 <= idx <= max_idx:
                print(f"Out of range — pick a number between 0 and {max_idx}.")
                continue
            return legal_actions[idx]

    @staticmethod
    def _default_render_state(obs: ObservableState) -> str:
        return f"State (you = player {obs.player_id}):\n  {obs.data}"

    @staticmethod
    def _default_render_action(action: Action) -> str:
        return str(action.data)
