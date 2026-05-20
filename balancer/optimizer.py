from __future__ import annotations

from typing import Any, Callable

import optuna

from core.base_adapter import BaseAdapter
from core.base_agent import BaseAgent
from core.engine import SimulationEngine
from core.stats import StatsCollector
from core.types import GameResult
from rl.mcts_agent import MCTSAgent


# Param-space entry shape: ("int", lo, hi) | ("float", lo, hi) | ("cat", [choices])
# Adapter factory: Callable[[dict[str, Any]], BaseAdapter]
# Agent factory:   Callable[[BaseAdapter, int], list[BaseAgent]]


class BalanceOptimizer:
    """Optuna-driven game-parameter tuner.

    Wraps an Optuna study that searches `param_space` for adapter configurations
    minimizing a penalty against the supplied `balance_targets`. Each trial:

      1. Sample a parameter dict from `param_space`.
      2. Build an adapter via `adapter_factory(params)`.
      3. Play `n_games_per_trial` games with `agent_factory(adapter, n_players)`
         (defaults to MCTS-vs-MCTS with `mcts_simulations` rollouts).
      4. Aggregate results with `StatsCollector`.
      5. Compute `balance_score` — distance from every supplied target. Lower
         is better; 0 means every target is satisfied.

    `param_space` schema:
        {"name": ("int",   low, high)}     -> trial.suggest_int
        {"name": ("float", low, high)}     -> trial.suggest_float
        {"name": ("cat",   [a, b, c])}     -> trial.suggest_categorical

    `balance_targets` keys (all optional, all contribute additively to the
    penalty if violated):
        "win_rate_range":   (lo, hi)  per-player win rate must fall in [lo, hi]
        "avg_turns_range":  (lo, hi)  average game length must fall in [lo, hi]
        "max_illegal_rate": float     per-player illegal-action rate ceiling
    """

    def __init__(
        self,
        adapter_factory: Callable[[dict[str, Any]], BaseAdapter],
        param_space: dict[str, tuple],
        balance_targets: dict[str, Any],
        n_trials: int = 50,
        n_games_per_trial: int = 30,
        agent_factory: Callable[[BaseAdapter, int], list[BaseAgent]] | None = None,
        mcts_simulations: int = 50,
        max_turns: int = 1000,
        seed: int | None = None,
    ) -> None:
        self.adapter_factory = adapter_factory
        self.param_space = param_space
        self.balance_targets = balance_targets
        self.n_trials = n_trials
        self.n_games_per_trial = n_games_per_trial
        self.mcts_simulations = mcts_simulations
        self.max_turns = max_turns
        self.seed = seed
        self.agent_factory = agent_factory or self._default_agent_factory
        self.study: optuna.Study | None = None

    def optimize(self) -> dict[str, Any]:
        """Run the study and return the best parameter set."""
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=False,
        )
        return dict(self.study.best_params)

    def print_best_params(self) -> None:
        """Pretty-print the best trial's score and parameters."""
        if self.study is None:
            print("No optimization run yet. Call optimize() first.")
            return
        print(f"=== Best balance score: {self.study.best_value:.4f} ===")
        print("Best parameters:")
        for key, value in self.study.best_params.items():
            print(f"  {key} = {value}")

    def plot_optimization_history(self, output_path: str | None = None):
        """Render a matplotlib convergence plot. Saves to `output_path` if given.

        Returns the matplotlib Axes for further tweaking.
        """
        if self.study is None:
            raise RuntimeError("Call optimize() first.")
        from optuna.visualization.matplotlib import plot_optimization_history

        ax = plot_optimization_history(self.study)
        if output_path is not None:
            ax.figure.savefig(output_path)
        return ax

    def plot_param_importances(self, output_path: str | None = None):
        """Render a matplotlib parameter-importance chart. Saves to `output_path` if given.

        Requires scikit-learn (pip install scikit-learn). Returns None if unavailable.
        Returns the matplotlib Axes for further tweaking.
        """
        if self.study is None:
            raise RuntimeError("Call optimize() first.")
        try:
            from optuna.visualization.matplotlib import plot_param_importances
        except ImportError:
            print(
                "plot_param_importances ignoré : scikit-learn manquant "
                "(pip install scikit-learn)."
            )
            return None
        try:
            ax = plot_param_importances(self.study)
        except ImportError:
            print(
                "plot_param_importances ignoré : scikit-learn manquant "
                "(pip install scikit-learn)."
            )
            return None
        if output_path is not None:
            ax.figure.savefig(output_path)
        return ax

    # ------------------------------------------------------------------ internals

    def _default_agent_factory(self, adapter: BaseAdapter, n_players: int) -> list[BaseAgent]:
        return [
            MCTSAgent(
                adapter,
                n_simulations=self.mcts_simulations,
                seed=None if self.seed is None else self.seed + i,
            )
            for i in range(n_players)
        ]

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, spec in self.param_space.items():
            kind = spec[0]
            if kind == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2])
            elif kind == "float":
                params[name] = trial.suggest_float(name, spec[1], spec[2])
            elif kind == "cat":
                params[name] = trial.suggest_categorical(name, spec[1])
            else:
                raise ValueError(f"Unknown param spec kind {kind!r} for {name}")
        return params

    def _objective(self, trial: optuna.Trial) -> float:
        params = self._suggest_params(trial)
        adapter = self.adapter_factory(params)
        agents = self.agent_factory(adapter, adapter.get_n_players())
        engine = SimulationEngine(adapter, agents, max_turns=self.max_turns)
        results = [engine.run_game() for _ in range(self.n_games_per_trial)]
        return self._balance_score(results)

    def _balance_score(self, results: list[GameResult]) -> float:
        """Penalize every target violation; lower is better, 0 = perfect."""
        stats = StatsCollector(results)
        score = 0.0

        if "win_rate_range" in self.balance_targets:
            lo, hi = self.balance_targets["win_rate_range"]
            for wr in stats.win_rates().values():
                if wr < lo:
                    score += (lo - wr) * 10
                elif wr > hi:
                    score += (wr - hi) * 10

        if "avg_turns_range" in self.balance_targets:
            lo, hi = self.balance_targets["avg_turns_range"]
            t = stats.avg_turns()
            if t < lo:
                score += (lo - t) / max(lo, 1.0)
            elif t > hi:
                score += (t - hi) / max(hi, 1.0)

        if "max_illegal_rate" in self.balance_targets:
            max_rate = self.balance_targets["max_illegal_rate"]
            for rate in stats.illegal_action_rates().values():
                if rate > max_rate:
                    score += (rate - max_rate) * 5

        # Timeouts are always a defect — strongly penalize regardless of targets.
        timed_out_frac = stats.timed_out_count() / stats.n_games
        score += timed_out_frac * 10

        return score
