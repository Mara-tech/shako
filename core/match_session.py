from __future__ import annotations


class MatchSession:
    """Tracks round number, per-seat win tally, and first-player rotation
    across a replayed human-vs-agent match.

    Round-to-round bookkeeping is a caller concern (see CLAUDE.md) — this
    class holds no reference to any adapter, agent, or `GameResult`; it only
    aggregates outcomes the caller already computed. `record()` must be
    called for the round that was just played *before* `next_round()` or
    `reset()` advances `round_number`.
    """

    def __init__(self, n_players: int, human_seat: int) -> None:
        self.n_players = n_players
        self.human_seat = human_seat
        self.round_number = 1
        self.wins: dict[int, int] = {pid: 0 for pid in range(n_players)}
        self.draws = 0

    def _offset(self) -> int:
        return (self.round_number - 1) % self.n_players

    def rotate(self, seats: list) -> list:
        """Left-rotate `seats` so this round's player_id 0 is last round's seat `_offset()`."""
        o = self._offset()
        return seats[o:] + seats[:o]

    def human_player_id(self) -> int:
        return (self.human_seat - self._offset()) % self.n_players

    def record(self, winner_id: int | None) -> None:
        if winner_id is None:
            self.draws += 1
            return
        original_seat = (winner_id + self._offset()) % self.n_players
        self.wins[original_seat] += 1

    def next_round(self) -> None:
        self.round_number += 1

    def reset(self) -> None:
        self.round_number = 1
        self.wins = {pid: 0 for pid in range(self.n_players)}
        self.draws = 0
