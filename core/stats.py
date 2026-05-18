from __future__ import annotations

import statistics
from typing import Any

from core.types import GameResult


class StatsCollector:
    """Aggregates a list of `GameResult` objects into balancing-relevant metrics.

    All accessors are lazy (recomputed on every call). Construct a new
    `StatsCollector` if you append more results.
    """

    def __init__(self, results: list[GameResult]) -> None:
        if not results:
            raise ValueError("StatsCollector requires at least one GameResult")
        self.results = results
        seats: set[int] = set()
        for r in results:
            seats.update(r.scores.keys())
            seats.update(r.illegal_action_counts.keys())
        self._n_players = (max(seats) + 1) if seats else 0

    @property
    def n_games(self) -> int:
        return len(self.results)

    def win_rates(self) -> dict[int, float]:
        """Fraction of games each player won outright. Draws aren't credited."""
        wins = {pid: 0 for pid in range(self._n_players)}
        for r in self.results:
            if r.winner_id is not None:
                wins[r.winner_id] = wins.get(r.winner_id, 0) + 1
        return {pid: w / self.n_games for pid, w in wins.items()}

    def avg_duration(self) -> float:
        """Mean wall-clock duration of a game in milliseconds."""
        return statistics.mean(r.duration_ms for r in self.results)

    def avg_turns(self) -> float:
        """Mean number of turns per game."""
        return statistics.mean(r.n_turns for r in self.results)

    def score_distribution(self) -> dict[int, dict[str, float]]:
        """Per-player final-score statistics: mean, stdev, min, max."""
        dist: dict[int, dict[str, float]] = {}
        for pid in range(self._n_players):
            scores = [r.scores.get(pid, 0.0) for r in self.results]
            dist[pid] = {
                "mean": statistics.mean(scores),
                "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "min": min(scores),
                "max": max(scores),
            }
        return dist

    def illegal_action_rates(self) -> dict[int, float]:
        """Per-player rate of illegal actions, normalized by that player's turns.

        Assumes turns are split evenly among players (good approximation for
        most turn-based games). Returns illegal_count / estimated_turns_played.
        """
        totals = {pid: 0 for pid in range(self._n_players)}
        for r in self.results:
            for pid, count in r.illegal_action_counts.items():
                totals[pid] = totals.get(pid, 0) + count
        total_turns = sum(r.n_turns for r in self.results)
        if total_turns == 0 or self._n_players == 0:
            return {pid: 0.0 for pid in range(self._n_players)}
        per_player_turns = total_turns / self._n_players
        return {pid: totals[pid] / per_player_turns for pid in range(self._n_players)}

    def timed_out_count(self) -> int:
        """Number of games that hit `max_turns` and were force-terminated."""
        return sum(1 for r in self.results if r.timed_out)

    def summary(self) -> dict[str, Any]:
        """Return all statistics as a single nested dict — ready for JSON dump."""
        return {
            "n_games": self.n_games,
            "n_players": self._n_players,
            "win_rates": self.win_rates(),
            "avg_duration_ms": self.avg_duration(),
            "avg_turns": self.avg_turns(),
            "score_distribution": self.score_distribution(),
            "illegal_action_rates": self.illegal_action_rates(),
            "timed_out_games": self.timed_out_count(),
        }

    def print_report(self) -> None:
        """Print a human-readable summary to stdout."""
        s = self.summary()
        print(f"=== Shako Stats Report ({s['n_games']} games) ===")
        print(f"Avg duration : {s['avg_duration_ms']:.1f} ms")
        print(f"Avg turns    : {s['avg_turns']:.1f}")
        print(f"Timed out    : {s['timed_out_games']}")
        print()
        print("Player | Win rate | Score mean ± std  |   Min /   Max | Illegal/turn")
        print("-" * 74)
        wins = s["win_rates"]
        scores = s["score_distribution"]
        illegals = s["illegal_action_rates"]
        for pid in sorted(wins.keys()):
            sd = scores[pid]
            print(
                f"  {pid:>4} | {wins[pid]:>7.2%} | "
                f"{sd['mean']:>7.2f} ± {sd['stdev']:>6.2f} | "
                f"{sd['min']:>6.2f} / {sd['max']:>6.2f} | "
                f"{illegals[pid]:>10.4f}"
            )
