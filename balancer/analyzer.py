from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any

from core.types import GameResult


Severity = str  # "low" | "medium" | "high" | "critical"


@dataclass
class Issue:
    """A single balance pathology detected in a batch of games."""

    category: str
    severity: Severity
    description: str
    metric: float | None = None


_SEVERITY_ORDER: dict[str, int] = {"critical": 0, "high": 1, "medium": 2, "low": 3}


class DominanceAnalyzer:
    """Inspects a batch of `GameResult`s for common balance pathologies.

    Detectors:
      - `detect_seat_advantage` — first/last player wins disproportionately.
      - `detect_low_action_entropy` — an agent collapses onto a tiny set of
        actions (often a sign that one strategy strictly dominates).
      - `detect_rare_actions` — game-design surface area that almost never
        sees play (likely strict dominance the other way).
      - `detect_runaway_duration` — games that hit `max_turns` or whose length
        variance suggests no clear convergence to terminal states.

    All "actions" detectors require the GameResults to have `actions != None`
    (i.e. the engine was run with `record=True`). Missing recordings are
    silently skipped — the report still surfaces seat-advantage and runaway
    issues, just without action-level findings.
    """

    def __init__(self, results: list[GameResult]) -> None:
        if not results:
            raise ValueError("DominanceAnalyzer requires at least one GameResult")
        self.results = results
        seats: set[int] = set()
        for r in results:
            seats.update(r.scores.keys())
        self._n_players = (max(seats) + 1) if seats else 0
        self._has_actions = all(r.actions is not None for r in results)

    # ------------------------------------------------------------------ detectors

    def detect_seat_advantage(self) -> Issue | None:
        wins = {pid: 0 for pid in range(self._n_players)}
        decided = 0
        for r in self.results:
            if r.winner_id is not None:
                wins[r.winner_id] = wins.get(r.winner_id, 0) + 1
                decided += 1
        if decided == 0:
            return None
        rates = {pid: w / decided for pid, w in wins.items()}
        expected = 1.0 / self._n_players
        max_dev = max(abs(r - expected) for r in rates.values())
        if max_dev <= 0.08:
            return None
        severity: Severity
        if max_dev > 0.30:
            severity = "critical"
        elif max_dev > 0.15:
            severity = "high"
        else:
            severity = "medium"
        favored = max(rates, key=lambda k: rates[k])
        return Issue(
            category="seat_advantage",
            severity=severity,
            description=(
                f"player {favored} wins {rates[favored]:.0%} of decided games "
                f"(expected ~{expected:.0%}); max deviation {max_dev:.0%}"
            ),
            metric=max_dev,
        )

    def detect_low_action_entropy(self) -> list[Issue]:
        if not self._has_actions:
            return []
        per_player: dict[int, Counter[str]] = {pid: Counter() for pid in range(self._n_players)}
        for r in self.results:
            for _turn, pid, action in r.actions or []:
                per_player[pid][_action_key(action)] += 1

        issues: list[Issue] = []
        for pid, counter in per_player.items():
            total = sum(counter.values())
            if total < 10:
                continue  # insufficient sample
            n_distinct = len(counter)
            if n_distinct <= 1:
                issues.append(
                    Issue(
                        category="low_action_entropy",
                        severity="high",
                        description=(
                            f"player {pid} only ever picks one action "
                            f"({_truncate(next(iter(counter)), 60)})"
                        ),
                        metric=0.0,
                    )
                )
                continue
            probs = [c / total for c in counter.values()]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            max_entropy = math.log2(n_distinct)
            normalized = entropy / max_entropy if max_entropy > 0 else 1.0
            if normalized < 0.3:
                severity: Severity = "high"
            elif normalized < 0.5:
                severity = "medium"
            else:
                continue
            issues.append(
                Issue(
                    category="low_action_entropy",
                    severity=severity,
                    description=(
                        f"player {pid} action distribution concentrated: "
                        f"normalized entropy {normalized:.2f} over {n_distinct} distinct actions"
                    ),
                    metric=normalized,
                )
            )
        return issues

    def detect_rare_actions(self, threshold: float = 0.01) -> list[Issue]:
        if not self._has_actions:
            return []
        counter: Counter[str] = Counter()
        for r in self.results:
            for _turn, _pid, action in r.actions or []:
                counter[_action_key(action)] += 1
        total = sum(counter.values())
        if total == 0:
            return []
        issues: list[Issue] = []
        for key, count in counter.items():
            frac = count / total
            if frac < threshold:
                issues.append(
                    Issue(
                        category="rare_action",
                        severity="low",
                        description=(
                            f"action {_truncate(key, 50)} used in only "
                            f"{frac:.2%} of moves ({count}/{total})"
                        ),
                        metric=frac,
                    )
                )
        return issues

    def detect_runaway_duration(self) -> Issue | None:
        timed_out = sum(1 for r in self.results if r.timed_out)
        frac = timed_out / len(self.results)
        if frac > 0.20:
            return Issue(
                category="runaway_duration",
                severity="critical",
                description=f"{frac:.0%} of games hit max_turns and were force-terminated",
                metric=frac,
            )
        if frac > 0.05:
            return Issue(
                category="runaway_duration",
                severity="high",
                description=f"{frac:.0%} of games hit max_turns",
                metric=frac,
            )

        turns = [r.n_turns for r in self.results]
        if len(turns) < 2:
            return None
        mean = statistics.mean(turns)
        if mean <= 0:
            return None
        cv = statistics.stdev(turns) / mean
        if cv > 1.0:
            return Issue(
                category="runaway_duration",
                severity="high",
                description=(
                    f"game length variance very high "
                    f"(CV={cv:.2f}, mean={mean:.1f} turns)"
                ),
                metric=cv,
            )
        if cv > 0.5:
            return Issue(
                category="runaway_duration",
                severity="medium",
                description=(
                    f"game length variance elevated "
                    f"(CV={cv:.2f}, mean={mean:.1f} turns)"
                ),
                metric=cv,
            )
        return None

    # ------------------------------------------------------------------ aggregation

    def report(self) -> list[Issue]:
        """Run every detector and return all issues, sorted critical-first."""
        issues: list[Issue] = []
        seat = self.detect_seat_advantage()
        if seat is not None:
            issues.append(seat)
        issues.extend(self.detect_low_action_entropy())
        issues.extend(self.detect_rare_actions())
        runaway = self.detect_runaway_duration()
        if runaway is not None:
            issues.append(runaway)
        issues.sort(key=lambda i: (_SEVERITY_ORDER[i.severity], i.category))
        return issues

    def print_report(self) -> None:
        issues = self.report()
        header = f"=== Balance Analysis ({len(self.results)} games)"
        if not issues:
            print(f"{header} — no issues detected ===")
            return
        print(f"{header} — {len(issues)} issue(s) ===")
        for issue in issues:
            print(f"  [{issue.severity.upper():>8}] {issue.category}: {issue.description}")


def _action_key(action: Any) -> str:
    return json.dumps(action.data, sort_keys=True, default=str)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"
