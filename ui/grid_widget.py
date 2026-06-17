from __future__ import annotations

from rich.text import Text
from textual import events
from textual.message import Message
from textual.widget import Widget

_CELL_W = 2  # chars per cell: symbol + trailing space


class GridWidget(Widget):
    """Clickable NxM board widget for Textual.

    Emits :class:`GridWidget.CellClicked` on mouse click.
    ``mode="column"`` maps any click to a column index (Connect Four style).
    ``mode="cell"`` maps a click to ``(row, col)`` (Tic-Tac-Toe style).

    Adapters control symbols and colours by passing dicts keyed on cell value
    (0 = empty, 1 = player 0, 2 = player 1, …).
    """

    DEFAULT_CSS = """
    GridWidget {
        height: auto;
        width: auto;
    }
    """

    class CellClicked(Message):
        """Posted when the user clicks a cell or column."""

        bubble = True

        def __init__(self, row: int, col: int) -> None:
            super().__init__()
            self.row = row
            self.col = col

    def __init__(
        self,
        rows: int,
        cols: int,
        mode: str = "cell",
        symbols: dict[int, str] | None = None,
        colors: dict[int, str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.rows = rows
        self.cols = cols
        self.mode = mode  # "column" or "cell"
        self._symbols = symbols or {0: "·", 1: "●", 2: "●"}
        self._colors = colors or {0: "dim", 1: "bold red", 2: "bold yellow"}
        self._board: list[int] = [0] * (rows * cols)
        self._hover_col: int = -1

    def set_board(self, board: list[int]) -> None:
        self._board = list(board)
        self.refresh()

    def _col_from_x(self, x: int) -> int:
        return min(max(x // _CELL_W, 0), self.cols - 1)

    def render(self) -> Text:
        parts: list[str] = []

        if self.mode == "column":
            row_parts = []
            for c in range(self.cols):
                mark = "▼" if c == self._hover_col else " "
                row_parts.append(f"[bold cyan]{mark}[/bold cyan] ")
            parts.append("".join(row_parts))

        for r in range(self.rows):
            row_parts = []
            for c in range(self.cols):
                val = self._board[r * self.cols + c]
                hover = self.mode == "column" and c == self._hover_col
                sym = self._symbols.get(val, "?")
                col = self._colors.get(val, "")
                if hover and val == 0:
                    row_parts.append(f"[{col} on blue]{sym}[/] ")
                else:
                    row_parts.append(f"[{col}]{sym}[/] ")
            parts.append("".join(row_parts))

        if self.mode == "column":
            parts.append("[dim]" + " ".join(str(c) for c in range(self.cols)) + "[/dim]")

        return Text.from_markup("\n".join(parts))

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self.mode == "column":
            col = self._col_from_x(event.x)
            if col != self._hover_col:
                self._hover_col = col
                self.refresh()

    def on_leave(self) -> None:
        self._hover_col = -1
        self.refresh()

    def on_click(self, event: events.Click) -> None:
        col = self._col_from_x(event.x)
        row_offset = 1 if self.mode == "column" else 0
        row = min(max(event.y - row_offset, 0), self.rows - 1)
        self.post_message(self.CellClicked(row, col))
