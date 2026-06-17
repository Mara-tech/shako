from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

from core.base_adapter import BaseAdapter
from core.base_agent import BaseAgent
from core.types import Action, ObservableState


class RichHumanAgent(BaseAgent):
    """Human agent with Rich-styled board display and coloured action list."""

    def __init__(self, adapter: BaseAdapter) -> None:
        self._adapter = adapter
        self._console = Console()
        self.player_id: int = 0

    def on_game_start(self, player_id: int, n_players: int) -> None:
        self.player_id = player_id
        self._console.rule(
            f"[bold cyan]New game — you are Player {player_id} (of {n_players})[/bold cyan]"
        )

    def on_game_end(self, scores: dict[int, float]) -> None:
        self._console.rule("[bold]Game over[/bold]")
        for pid, s in sorted(scores.items()):
            mark = "[bold green]★[/bold green]" if pid == self.player_id else "  "
            self._console.print(f"  {mark} Player {pid}: {s}")

    def choose_action(
        self, observable_state: ObservableState, legal_actions: list[Action]
    ) -> Action:
        self._console.print()
        renderable = self._adapter.get_rich_renderable(observable_state)
        if renderable is not None:
            self._console.print(
                Panel(
                    renderable,
                    title=f"[cyan]Player {observable_state.player_id}'s turn[/cyan]",
                    expand=False,
                )
            )
        else:
            self._console.print(f"[dim]State:[/dim] {observable_state.data}")

        offset = self._adapter.get_action_index_offset()
        self._console.print("\n[bold]Legal actions:[/bold]")
        for i, action in enumerate(legal_actions):
            display = _action_display(self._adapter, action)
            self._console.print(f"  [[cyan]{i + offset:>3}[/cyan]] {display}")

        lo, hi = offset, len(legal_actions) - 1 + offset
        while True:
            raw = input(f"Your move [{lo}-{hi}]: ").strip()
            if not raw:
                continue
            try:
                idx = int(raw)
            except ValueError:
                self._console.print(f"[red]Not a number: {raw!r}[/red]")
                continue
            if not lo <= idx <= hi:
                self._console.print(f"[red]Pick {lo} to {hi}.[/red]")
                continue
            return legal_actions[idx - offset]


def _action_display(adapter: BaseAdapter, action: Action) -> str:
    fn = getattr(adapter, "get_action_display", None)
    if fn is not None:
        return fn(action)
    return str(action.data)
