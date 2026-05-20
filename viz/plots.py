from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from core.types import GameResult


def plot_simulation_results(
    results: list[GameResult],
    game_name: str,
    output_path: str | Path | None = None,
) -> None:
    """Three-panel figure: win rates, score distributions, game lengths."""
    n = len(results)
    player_ids = sorted({pid for r in results for pid in r.scores})
    n_players = len(player_ids)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"{game_name} — {n} parties", fontweight="bold")

    # --- Win rate ---
    ax = axes[0]
    win_counts = {pid: 0 for pid in player_ids}
    for r in results:
        if r.winner_id is not None:
            win_counts[r.winner_id] += 1
    win_rates = [win_counts[pid] / n for pid in player_ids]
    # ±1.96 * sqrt(p(1-p)/n) confidence interval, clipped to [0, 1]
    errors = [
        min(1.96 * math.sqrt(max(p * (1 - p) / n, 0)), p, 1 - p)
        for p in win_rates
    ]
    bars = ax.bar([str(pid) for pid in player_ids], win_rates, yerr=errors,
                  capsize=5, color="steelblue", alpha=0.8)
    ax.axhline(1 / n_players, linestyle="--", color="gray", linewidth=1,
               label=f"équilibre ({1/n_players:.0%})")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Joueur")
    ax.set_title("Win rate")
    ax.legend(fontsize=8)

    # --- Score distribution ---
    ax = axes[1]
    score_data = [[r.scores[pid] for r in results if pid in r.scores] for pid in player_ids]
    bp = ax.boxplot(score_data, labels=[str(pid) for pid in player_ids], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    ax.set_xlabel("Joueur")
    ax.set_title("Distribution des scores")

    # --- Game lengths ---
    ax = axes[2]
    turns = [r.n_turns for r in results]
    ax.hist(turns, bins=min(30, n), color="steelblue", alpha=0.8, edgecolor="white")
    median = sorted(turns)[len(turns) // 2]
    ax.axvline(median, linestyle="--", color="gray", linewidth=1,
               label=f"médiane = {median}")
    timed_out = sum(1 for r in results if r.timed_out)
    if timed_out:
        ax.set_title(f"Longueur des parties ({timed_out} timeouts)")
    else:
        ax.set_title("Longueur des parties")
    ax.set_xlabel("Nombre de tours")
    ax.legend(fontsize=8)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_selfplay_history(
    history: list[dict[str, Any]],
    promotion_threshold: float = 0.55,
    output_path: str | Path | None = None,
) -> None:
    """Win-rate-per-iteration curve with promotion markers."""
    if not history:
        return

    iterations = [h["iteration"] + 1 for h in history]
    win_rates = [h["candidate_win_rate"] for h in history]
    promoted = [h["promoted"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iterations, win_rates, color="steelblue", linewidth=1.5, zorder=2)
    for it, wr, prom in zip(iterations, win_rates, promoted):
        ax.scatter(it, wr, color="green" if prom else "red",
                   s=60, zorder=3, linewidths=0)
    ax.axhline(promotion_threshold, linestyle="--", color="gray", linewidth=1,
               label=f"seuil de promotion ({promotion_threshold:.0%})")
    ax.scatter([], [], color="green", s=60, label="promu")
    ax.scatter([], [], color="red", s=60, label="rejeté")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Itération")
    ax.set_title("Self-play — win rate du candidat")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
