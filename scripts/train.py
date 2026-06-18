#!/usr/bin/env python
"""Non-interactive self-play training script.

Usage:
    python scripts/train.py --game nim --n-iterations 10 --seed 0
    python scripts/train.py --game connect4 --adapter-rows 6 --adapter-cols 7
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import re
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Union, get_args, get_origin

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_GAMES_DIR = _ROOT / "games"
_SKIP_FROM_PATH: set[str] = {"seed"}


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


def _model_dir(game_name: str, params: dict[str, Any]) -> Path:
    parts = [f"{k}={v}" for k, v in params.items() if k not in _SKIP_FROM_PATH]
    base = _GAMES_DIR / game_name / "models" / "selfplay"
    return base.joinpath(*parts) if parts else base


def _parse_bool(val: str) -> bool:
    return val.lower() not in ("false", "0", "no", "")


def _annotation_to_argtype(annotation: Any) -> Callable[[str], Any]:
    """Map a type annotation (string or type object) to an argparse type callable."""
    _str_map: dict[str, Callable[[str], Any]] = {
        "int": int, "float": float, "bool": _parse_bool, "str": str,
    }
    _type_map: dict[Any, Callable[[str], Any]] = {
        int: int, float: float, bool: _parse_bool, str: str,
    }
    if annotation is inspect.Parameter.empty:
        return str
    if isinstance(annotation, str):
        bare = annotation.strip()
        bare = re.sub(r"^Optional\[(.+)]$", r"\1", bare).strip()
        bare = re.sub(r"\s*\|\s*None\b|\bNone\s*\|\s*", "", bare).strip()
        return _str_map.get(bare, str)
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Union or (
        hasattr(types, "UnionType") and isinstance(annotation, types.UnionType)
    ):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            annotation = non_none[0]
    return _type_map.get(annotation, str)


def _add_adapter_args(parser: argparse.ArgumentParser, adapter_class: type) -> list[str]:
    """Add adapter __init__ params as --adapter-<name> CLI args; return param names."""
    sig = inspect.signature(adapter_class)
    names: list[str] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        names.append(name)
        has_default = param.default is not inspect.Parameter.empty
        parser.add_argument(
            f"--adapter-{name.replace('_', '-')}",
            dest=f"adapter_{name}",
            type=_annotation_to_argtype(param.annotation),
            default=param.default if has_default else None,
            required=not has_default,
            metavar=name.upper(),
        )
    return names


def main() -> None:
    # First pass: resolve --game before building the full parser.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--game", required=True)
    known, _ = pre.parse_known_args()

    adapter_class = _import_adapter_class(known.game)

    parser = argparse.ArgumentParser(description="Non-interactive self-play trainer")
    parser.add_argument("--game", required=True, help="Game name (folder under games/)")
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--n-games-per-iter", type=int, default=50)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--mcts-simulations", type=int, default=100)
    parser.add_argument("--promotion-threshold", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (omit for random)")

    adapter_param_names = _add_adapter_args(parser, adapter_class)
    args = parser.parse_args()

    adapter_params: dict[str, Any] = {
        name: getattr(args, f"adapter_{name}")
        for name in adapter_param_names
        if getattr(args, f"adapter_{name}") is not None
    }

    try:
        adapter = adapter_class(**adapter_params)
    except TypeError as exc:
        raise SystemExit(
            f"{adapter_class.__name__} requires constructor arguments: {exc}\n"
            "Pass them with --adapter-<param-name> <value>."
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
    out_dir = _model_dir(args.game, adapter_params)
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
