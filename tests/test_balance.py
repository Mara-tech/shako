from __future__ import annotations

from balancer.optimizer import BalanceOptimizer
from core.base_adapter import BaseAdapter
from games.nim.adapter import NimAdapter
from rl.random_agent import RandomAgent


def _nim_factory(params: dict) -> NimAdapter:
    return NimAdapter(n_sticks=params["n_sticks"], max_take=3, last_takes_wins=False)


def _random_agents(adapter: BaseAdapter, n_players: int) -> list:
    # Random vs random keeps trials fast and makes win rates noisy enough that
    # different n_sticks values produce different penalties — without that
    # variance, Optuna has nothing to learn from.
    return [RandomAgent(seed=i * 17 + 1) for i in range(n_players)]


def test_optimizer_searches_and_returns_a_best_within_param_bounds() -> None:
    optimizer = BalanceOptimizer(
        adapter_factory=_nim_factory,
        param_space={"n_sticks": ("int", 5, 25)},
        balance_targets={"win_rate_range": (0.40, 0.60)},
        n_trials=12,
        n_games_per_trial=15,
        agent_factory=_random_agents,
        seed=0,
    )
    best = optimizer.optimize()

    assert "n_sticks" in best
    assert 5 <= best["n_sticks"] <= 25
    assert optimizer.study is not None
    assert len(optimizer.study.trials) == 12


def test_optimizer_best_score_improves_versus_search_baseline() -> None:
    """The best trial value should beat the population's average — TPE should
    surface lower-penalty configurations than random sampling alone.

    We check `best < mean` (not `best < first`) because TPE's first few trials
    are random samples, and "improvement over the first trial" is too noisy
    a claim with this many random-vs-random games.
    """
    import statistics

    optimizer = BalanceOptimizer(
        adapter_factory=_nim_factory,
        param_space={"n_sticks": ("int", 5, 25)},
        balance_targets={"win_rate_range": (0.40, 0.60)},
        n_trials=12,
        n_games_per_trial=15,
        agent_factory=_random_agents,
        seed=0,
    )
    optimizer.optimize()

    assert optimizer.study is not None
    values = [t.value for t in optimizer.study.trials if t.value is not None]
    assert optimizer.study.best_value == min(values)
    assert optimizer.study.best_value < statistics.mean(values), (
        f"best={optimizer.study.best_value:.4f} not better than "
        f"mean={statistics.mean(values):.4f} — search did not improve"
    )

    # And Optuna actually explored a few distinct points (otherwise it's stuck).
    distinct = len({round(v, 4) for v in values})
    assert distinct >= 3, f"only {distinct} distinct trial values across {len(values)} trials"
