from __future__ import annotations

from balancer.analyzer import DominanceAnalyzer
from balancer.optimizer import BalanceOptimizer
from core.types import Action, GameResult
from games.nim.adapter import NimAdapter


def _mk_result(
    winner: int | None,
    scores: dict[int, float],
    n_turns: int = 10,
    actions: list[tuple[int, int, Action]] | None = None,
    timed_out: bool = False,
) -> GameResult:
    return GameResult(
        scores=scores,
        n_turns=n_turns,
        winner_id=winner,
        duration_ms=10.0,
        actions=actions,
        timed_out=timed_out,
    )


# -------- DominanceAnalyzer --------------------------------------------------


def test_analyzer_flags_seat_advantage_when_one_player_always_wins() -> None:
    results = [_mk_result(winner=0, scores={0: 1.0, 1: 0.0}) for _ in range(50)]
    analyzer = DominanceAnalyzer(results)
    issue = analyzer.detect_seat_advantage()
    assert issue is not None
    assert issue.severity == "critical"
    assert "player 0" in issue.description


def test_analyzer_silent_on_balanced_seat_distribution() -> None:
    results = [_mk_result(winner=i % 2, scores={0: 1.0, 1: 0.0}) for i in range(50)]
    analyzer = DominanceAnalyzer(results)
    assert analyzer.detect_seat_advantage() is None


def test_analyzer_flags_collapsed_action_distribution() -> None:
    # Player 0 always plays {"take": 1}; player 1 plays varied actions.
    p0_action = Action(data={"take": 1})
    varied = [Action(data={"take": k}) for k in (1, 2, 3)]
    results = []
    for game_idx in range(20):
        actions = []
        for turn in range(6):
            pid = turn % 2
            act = p0_action if pid == 0 else varied[(game_idx + turn) % 3]
            actions.append((turn, pid, act))
        results.append(_mk_result(
            winner=game_idx % 2,
            scores={0: float(game_idx % 2 == 0), 1: float(game_idx % 2 == 1)},
            n_turns=6,
            actions=actions,
        ))
    analyzer = DominanceAnalyzer(results)
    issues = analyzer.detect_low_action_entropy()
    # Player 0 collapsed onto one action -> flagged; player 1 varied -> not flagged.
    flagged_players = {int(i.description.split()[1]) for i in issues}
    assert 0 in flagged_players
    assert 1 not in flagged_players


def test_analyzer_flags_runaway_when_games_time_out() -> None:
    results = [_mk_result(winner=None, scores={0: 0.0, 1: 0.0}, timed_out=True)
               for _ in range(10)]
    analyzer = DominanceAnalyzer(results)
    issue = analyzer.detect_runaway_duration()
    assert issue is not None
    assert issue.severity == "critical"


def test_analyzer_report_sorts_critical_first() -> None:
    # Mix: seat advantage (critical) + runaway (medium) via length variance.
    results = []
    for i in range(50):
        # Player 0 always wins -> critical seat issue.
        n = 5 if i % 2 == 0 else 50  # high variance -> medium runaway
        results.append(_mk_result(winner=0, scores={0: 1.0, 1: 0.0}, n_turns=n))
    analyzer = DominanceAnalyzer(results)
    report = analyzer.report()
    assert len(report) >= 2
    assert report[0].severity == "critical"


# -------- BalanceOptimizer ---------------------------------------------------


def test_optimizer_runs_and_returns_best_params() -> None:
    """Smoke test: search n_sticks for misère Nim within a tiny space.

    The optimizer should at least converge on parameter dict keys that match
    the param_space, and `study.best_value` should be defined.
    """

    def adapter_factory(params: dict) -> NimAdapter:
        return NimAdapter(n_sticks=params["n_sticks"], max_take=3, last_takes_wins=False)

    optimizer = BalanceOptimizer(
        adapter_factory=adapter_factory,
        param_space={"n_sticks": ("int", 5, 12)},
        balance_targets={
            "win_rate_range": (0.40, 0.60),
            "avg_turns_range": (4, 20),
        },
        n_trials=5,
        n_games_per_trial=6,
        mcts_simulations=15,
        seed=0,
    )
    best = optimizer.optimize()
    assert "n_sticks" in best
    assert 5 <= best["n_sticks"] <= 12
    assert optimizer.study is not None
    assert optimizer.study.best_value is not None
