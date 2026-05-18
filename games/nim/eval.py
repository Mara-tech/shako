from __future__ import annotations

from typing import Callable

from core.types import ObservableState


EvalFn = Callable[[ObservableState], float]


def _nim_score(state: ObservableState, max_take: int, last_takes_wins: bool) -> float:
    """Return 1.0 if `state` is a losing position for the player to move, else 0.0.

    A position is losing for the player about to move iff every move leaves the
    opponent in a winning position. For single-pile Nim with max_take = m, the
    closed form is `sticks % (m+1) == r` where r = 0 for normal play and r = 1
    for misère.

    GreedyAgent maximizes this evaluation, so it will steer the game toward
    states that are losing for *its opponent* — i.e. winning for itself.
    """
    sticks = int(state.data["sticks"])
    losing_residue = 0 if last_takes_wins else 1
    return 1.0 if (sticks % (max_take + 1)) == losing_residue else 0.0


def nim_eval(state: ObservableState) -> float:
    """Optimal evaluator for the default `NimAdapter` (max_take=3, misère).

    Use `make_nim_eval(...)` if your adapter uses different parameters.
    """
    return _nim_score(state, max_take=3, last_takes_wins=False)


def make_nim_eval(max_take: int = 3, last_takes_wins: bool = False) -> EvalFn:
    """Build an evaluator matching a non-default `NimAdapter` configuration."""

    def fn(state: ObservableState) -> float:
        return _nim_score(state, max_take=max_take, last_takes_wins=last_takes_wins)

    return fn
