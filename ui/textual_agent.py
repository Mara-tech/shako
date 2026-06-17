from __future__ import annotations

import queue
from typing import TYPE_CHECKING

from core.base_agent import BaseAgent
from core.types import Action, ObservableState

if TYPE_CHECKING:
    from ui.textual_app import ShakTUIApp


class TextualHumanAgent(BaseAgent):
    """Human agent bridging the synchronous engine thread to the Textual event loop.

    The engine runs in a daemon thread. ``choose_action`` posts a UI-update via
    ``call_from_thread`` then blocks on a thread-safe queue until the user picks
    an action in the TUI.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[Action] = queue.Queue()
        self.app: ShakTUIApp | None = None
        self.player_id: int = 0

    def reset(self) -> None:
        """Drain the queue before starting a new game."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def on_game_start(self, player_id: int, n_players: int) -> None:
        self.player_id = player_id
        if self.app is not None:
            self.app.call_from_thread(self.app.on_game_start_ui, player_id, n_players)

    def on_game_end(self, scores: dict[int, float]) -> None:
        if self.app is not None:
            self.app.call_from_thread(self.app.on_game_end_ui, scores)

    def choose_action(
        self, observable_state: ObservableState, legal_actions: list[Action]
    ) -> Action:
        if self.app is not None:
            self.app.call_from_thread(
                self.app.update_state_ui, observable_state, legal_actions
            )
        return self._queue.get()

    def post_action(self, action: Action) -> None:
        self._queue.put(action)
