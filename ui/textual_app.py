from __future__ import annotations

import copy
import threading
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static

from core.base_adapter import BaseAdapter
from core.engine import SimulationEngine
from core.match_session import MatchSession
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
        Binding("n", "next_round", "Next round"),
        Binding("m", "new_match", "New match"),
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
        self._game_over = False
        self._session = MatchSession(adapter.get_n_players(), human_seat)

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
        self._game_over = False
        n = self._adapter.get_n_players()
        seats = [
            self._agent if pid == self._human_seat else copy.deepcopy(self._bot_agent)
            for pid in range(n)
        ]
        engine = SimulationEngine(
            self._adapter, self._session.rotate(seats), record=False, max_turns=1000
        )

        threading.Thread(target=engine.run_game, daemon=True).start()

    def _tally_str(self) -> str:
        wins = "  ".join(f"P{s}:{w}" for s, w in sorted(self._session.wins.items()))
        return f"{wins}  Draws:{self._session.draws}"

    # ---- callbacks from engine thread via call_from_thread ----

    def on_game_start_ui(self, player_id: int, n_players: int) -> None:
        self.sub_title = f"Round {self._session.round_number} | {self._tally_str()} | You = Player {player_id}"
        self.query_one("#status", Static).update(
            f"[dim]Round {self._session.round_number} started. "
            f"You are Player {player_id} of {n_players}.[/dim]"
        )

    def _render_board(self, obs_state: ObservableState) -> None:
        if self._grid_config:
            from ui.grid_widget import GridWidget

            info: dict[str, Any] = getattr(self._adapter, "get_grid_info")(obs_state)
            self.query_one("#grid", GridWidget).set_board(info["board"])
        else:
            renderable = self._adapter.get_rich_renderable(obs_state)
            self.query_one("#board", Static).update(
                renderable if renderable is not None else str(obs_state.data)
            )

    def refresh_board_ui(self, obs_state: ObservableState) -> None:
        """Repaint the board only, without touching input state or the action list.

        Used for moves this app isn't waiting on input for (the human's own
        move just after submission, and every bot/opponent move), so the
        board never lags behind what the engine has already applied.
        """
        self._render_board(obs_state)

    def update_state_ui(
        self, obs_state: ObservableState, legal_actions: list[Action]
    ) -> None:
        self._legal_actions = legal_actions
        self._accepting_input = True

        self._render_board(obs_state)

        offset = self._adapter.get_action_index_offset()
        actions_widget = self.query_one("#actions", ListView)
        actions_widget.clear()
        for i, action in enumerate(legal_actions):
            actions_widget.append(
                ListItem(Label(f"[{i + offset:>2}] {_action_display(self._adapter, action)}"))
            )

        self.query_one("#status", Static).update(
            "[bold green]Your turn — choose an action below or click the board[/bold green]"
        )
        actions_widget.focus()

    def on_game_end_ui(self, scores: dict[int, float]) -> None:
        self._accepting_input = False
        self._game_over = True
        self.query_one("#actions", ListView).clear()

        top_score = max(scores.values())
        winners = [p for p, s in scores.items() if s == top_score]
        winner_id = winners[0] if len(winners) == 1 else None
        human_pid = self._session.human_player_id()

        if winner_id is None:
            msg = "[yellow]Draw![/yellow]"
        elif winner_id == human_pid:
            msg = "[bold green]You won![/bold green]"
        else:
            msg = f"[bold red]Player {winner_id} won.[/bold red]"

        self._session.record(winner_id)

        score_str = "  ".join(f"P{p}: {s}" for p, s in sorted(scores.items()))
        self.query_one("#status", Static).update(
            f"{msg}  {score_str}\n"
            f"[dim]Round {self._session.round_number} tally — {self._tally_str()}[/dim]\n"
            "[dim]Press [bold]n[/bold] for next round, [bold]m[/bold] for new match, "
            "or [bold]q[/bold] to quit.[/dim]"
        )
        self.sub_title = f"Round {self._session.round_number} | {self._tally_str()}"

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

    def _reset_board_widgets(self) -> None:
        if self._grid_config:
            from ui.grid_widget import GridWidget

            cfg = self._grid_config
            self.query_one("#grid", GridWidget).set_board(
                [0] * (cfg["rows"] * cfg["cols"])
            )
        else:
            self.query_one("#board", Static).update("")
        self.query_one("#actions", ListView).clear()

    def action_next_round(self) -> None:
        if self._accepting_input or not self._game_over:
            return
        self._session.next_round()
        self._reset_board_widgets()
        self.query_one("#status", Static).update("[dim]Starting next round…[/dim]")
        self._start_engine_thread()

    def action_new_match(self) -> None:
        if self._accepting_input or not self._game_over:
            return
        self._session.reset()
        self._reset_board_widgets()
        self.query_one("#status", Static).update("[dim]Starting new match…[/dim]")
        self._start_engine_thread()

    def action_quit_game(self) -> None:
        self.exit()
