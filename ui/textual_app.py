from __future__ import annotations

import copy
import threading
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static

from core.base_adapter import BaseAdapter
from core.engine import SimulationEngine
from core.types import Action, ObservableState
from ui.textual_agent import TextualHumanAgent


def _action_display(adapter: BaseAdapter, action: Action) -> str:
    fn = getattr(adapter, "get_action_display", None)
    return fn(action) if fn is not None else str(action.data)


class ShakTUIApp(App):
    """Textual TUI for playing a Shako game as a human player."""

    CSS = """
    #grid, #board {
        border: round cyan;
        padding: 0 1;
        height: auto;
        margin: 0 0 1 0;
    }
    #status {
        padding: 0 1;
        height: 3;
        border: round $primary;
        margin: 0 0 1 0;
    }
    #actions {
        height: auto;
        max-height: 15;
        border: round yellow;
    }
    """

    BINDINGS = [
        Binding("q", "quit_game", "Quit"),
        Binding("r", "replay", "Play again"),
    ]

    def __init__(
        self,
        adapter: BaseAdapter,
        agent: TextualHumanAgent,
        bot_agent: Any,
        human_seat: int,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._agent = agent
        self._bot_agent = bot_agent
        self._human_seat = human_seat
        self._legal_actions: list[Action] = []
        self._accepting_input = False

        self._grid_config: dict | None = getattr(adapter, "get_grid_config", lambda: None)()
        agent.app = self

    def compose(self) -> ComposeResult:
        yield Header()
        if self._grid_config:
            from ui.grid_widget import GridWidget

            cfg = self._grid_config
            render_cfg: dict[str, Any] = getattr(self._adapter, "get_grid_render_config", lambda: {})()
            yield GridWidget(
                cfg["rows"],
                cfg["cols"],
                cfg["mode"],
                symbols=render_cfg.get("symbols"),
                colors=render_cfg.get("colors"),
                id="grid",
            )
        else:
            yield Static("", id="board")
        yield Static("[dim]Starting game…[/dim]", id="status")
        yield ListView(id="actions")
        yield Footer()

    def on_mount(self) -> None:
        game_name = type(self._adapter).__name__.replace("Adapter", "")
        self.title = f"Shako — {game_name}"
        self._start_engine_thread()

    def _start_engine_thread(self) -> None:
        self._agent.reset()
        n = self._adapter.get_n_players()
        agents = [
            self._agent if pid == self._human_seat else copy.deepcopy(self._bot_agent)
            for pid in range(n)
        ]
        engine = SimulationEngine(self._adapter, agents, record=False, max_turns=1000)

        threading.Thread(target=engine.run_game, daemon=True).start()

    # ---- callbacks from engine thread via call_from_thread ----

    def on_game_start_ui(self, player_id: int, n_players: int) -> None:
        self.sub_title = f"You = Player {player_id}"
        self.query_one("#status", Static).update(
            f"[dim]Game started. You are Player {player_id} of {n_players}.[/dim]"
        )

    def update_state_ui(
        self, obs_state: ObservableState, legal_actions: list[Action]
    ) -> None:
        self._legal_actions = legal_actions
        self._accepting_input = True

        if self._grid_config:
            from ui.grid_widget import GridWidget

            info: dict[str, Any] = getattr(self._adapter, "get_grid_info")(obs_state)
            self.query_one("#grid", GridWidget).set_board(info["board"])
        else:
            renderable = self._adapter.get_rich_renderable(obs_state)
            self.query_one("#board", Static).update(
                renderable if renderable is not None else str(obs_state.data)
            )

        actions_widget = self.query_one("#actions", ListView)
        actions_widget.clear()
        for i, action in enumerate(legal_actions):
            actions_widget.append(
                ListItem(Label(f"[{i:>2}] {_action_display(self._adapter, action)}"))
            )

        self.query_one("#status", Static).update(
            "[bold green]Your turn — choose an action below or click the board[/bold green]"
        )
        actions_widget.focus()

    def on_game_end_ui(self, scores: dict[int, float]) -> None:
        self._accepting_input = False
        self.query_one("#actions", ListView).clear()

        hs = self._human_seat
        top_score = max(scores.values())
        winners = [p for p, s in scores.items() if s == top_score]

        if len(winners) > 1:
            msg = "[yellow]Draw![/yellow]"
        elif hs in winners:
            msg = "[bold green]You won![/bold green]"
        else:
            msg = f"[bold red]Player {winners[0]} won.[/bold red]"

        score_str = "  ".join(f"P{p}: {s}" for p, s in sorted(scores.items()))
        self.query_one("#status", Static).update(
            f"{msg}  {score_str}\n"
            "[dim]Press [bold]r[/bold] to play again or [bold]q[/bold] to quit.[/dim]"
        )

    # ---- user input handlers ----

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if not self._accepting_input:
            return
        idx = event.list_view.index
        if idx is not None and 0 <= idx < len(self._legal_actions):
            self._submit_action(self._legal_actions[idx])

    def on_grid_widget_cell_clicked(self, event) -> None:
        if not self._accepting_input:
            return
        action = getattr(self._adapter, "get_action_for_click", lambda *_: None)(
            event.row, event.col, self._legal_actions
        )
        if action is not None:
            self._submit_action(action)

    def _submit_action(self, action: Action) -> None:
        self._accepting_input = False
        self.query_one("#status", Static).update("[dim]Bot is thinking…[/dim]")
        self._agent.post_action(action)

    # ---- key bindings ----

    def action_replay(self) -> None:
        if self._accepting_input:
            return
        if self._grid_config:
            from ui.grid_widget import GridWidget

            cfg = self._grid_config
            self.query_one("#grid", GridWidget).set_board(
                [0] * (cfg["rows"] * cfg["cols"])
            )
        else:
            self.query_one("#board", Static).update("")
        self.query_one("#status", Static).update("[dim]Starting new game…[/dim]")
        self.query_one("#actions", ListView).clear()
        self._start_engine_thread()

    def action_quit_game(self) -> None:
        self.exit()
