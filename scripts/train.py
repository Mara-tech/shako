#!/usr/bin/env python
"""Non-interactive self-play training script.

Usage:
    python scripts/train.py --game nim --n-iterations 10 --seed 0
"""
from __future__ import annotations

import argparse
import importlib
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_GAMES_DIR = _ROOT / "games"


def _import_adapter_class(name: str) -> type:
    from core.base_adapter import BaseAdapter

    module_path = f"games.{name}.adapter"
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise SystemExit(f"Cannot import {module_path}: {exc}") from exc

    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if isinstance(obj, type) and issubclass(obj, BaseAdapter) and obj is not BaseAdapter:
            return obj
    raise SystemExit(f"No BaseAdapter subclass found in {module_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-interactive self-play trainer")
    parser.add_argument("--game", required=True, help="Game name (folder under games/)")
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--n-games-per-iter", type=int, default=50)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--mcts-simulations", type=int, default=100)
    parser.add_argument("--promotion-threshold", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (omit for random)")
    args = parser.parse_args()

    adapter_class = _import_adapter_class(args.game)
    try:
        adapter = adapter_class()
    except TypeError as exc:
        raise SystemExit(
            f"{adapter_class.__name__} requires constructor arguments that cannot be "
            f"inferred automatically: {exc}\n"
            "Extend scripts/train.py to pass adapter params explicitly."
        ) from exc

    from rl.self_play import SelfPlayTrainer

    trainer = SelfPlayTrainer(
        adapter,
        n_iterations=args.n_iterations,
        n_games_per_iter=args.n_games_per_iter,
        eval_games=args.eval_games,
        mcts_simulations=args.mcts_simulations,
        promotion_threshold=args.promotion_threshold,
        verbose=True,
        seed=args.seed,
    )

    print(
        f"Training {args.game} — "
        f"{args.n_iterations} iter × {args.n_games_per_iter} games, "
        f"{args.mcts_simulations} MCTS sims, seed={args.seed}"
    )
    _, history = trainer.train()

    promotions = sum(1 for h in history if h["promoted"])
    print(f"\nDone. {promotions}/{args.n_iterations} iterations promoted the candidate.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _GAMES_DIR / args.game / "models" / "selfplay"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ts}.pkl"
    tmp = out_path.with_suffix(".pkl.tmp")
    try:
        trainer.save_agent(tmp)
        tmp.replace(out_path)
        print(f"Agent saved: {out_path.relative_to(_ROOT)}")
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Save failed: {exc}") from exc


if __name__ == "__main__":
    main()
