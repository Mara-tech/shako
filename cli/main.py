"""Shako interactive CLI.

Run with:
    python -m cli
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.syntax import Syntax
from rich.table import Table

from balancer.analyzer import DominanceAnalyzer
from balancer.optimizer import BalanceOptimizer
from core.base_adapter import BaseAdapter
from core.engine import SimulationEngine
from core.stats import StatsCollector
from core.types import GameResult
from rl.greedy_agent import GreedyAgent
from rl.mcts_agent import MCTSAgent
from rl.random_agent import RandomAgent
from rl.self_play import SelfPlayTrainer


_ROOT = Path(__file__).resolve().parent.parent
_GAMES_DIR = _ROOT / "games"

_SEVERITY_COLORS = {
    "critical": "bold red",
    "high": "yellow",
    "medium": "blue",
    "low": "dim",
}

console = Console()


def main() -> None:
    """Top-level interactive entry point."""
    console.rule("[bold cyan]Shako — Game Balancer[/bold cyan]")
    console.print(
        "[dim]Charge un jeu existant ou décris-en un nouveau ; "
        "ensuite configure la simulation, lance, et lis le rapport.[/dim]\n"
    )

    try:
        source = Prompt.ask(
            "Source du jeu",
            choices=["existing", "new"],
            default="existing",
        )
        if source == "new":
            game_name, adapter_class, param_space = _create_new_game()
        else:
            game_name, adapter_class = _load_existing_game()
            param_space = None

        if adapter_class is None:
            return

        adapter = _instantiate_safe(adapter_class)
        if adapter is None:
            return

        config = _configure_simulation()

        results = _run_simulation(adapter, game_name, config)

        best_params: dict[str, Any] | None = None
        if config["optimize"]:
            if param_space is None:
                param_space = _collect_parameter_space()
            if param_space:
                best_params = _run_optimization(adapter_class, param_space, config)

        _print_report(results, best_params, game_name)

        if best_params is not None:
            if Confirm.ask("\nRelancer la simulation avec ces paramètres ?", default=True):
                new_adapter = adapter_class(**best_params)
                results2 = _run_simulation(new_adapter, game_name, config)
                console.rule("[bold cyan]Re-run avec paramètres optimaux[/bold cyan]")
                _print_report(results2, None, game_name)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrompu.[/yellow]")


# -------------------------------------------------------------------- game loading


def _load_existing_game() -> tuple[str, type[BaseAdapter] | None]:
    available = sorted(
        d.name
        for d in _GAMES_DIR.iterdir()
        if d.is_dir() and (d / "adapter.py").exists()
    )
    if not available:
        console.print(f"[red]Aucun adapter trouvé sous {_GAMES_DIR}.[/red]")
        return "", None

    name = Prompt.ask("Quel jeu", choices=available, default=available[0])
    adapter_class = _import_adapter_class(name)
    return name, adapter_class


def _create_new_game() -> tuple[str, type[BaseAdapter] | None, dict[str, tuple]]:
    name = Prompt.ask("Nom du jeu (snake_case)").strip()
    if not name:
        console.print("[red]Nom requis.[/red]")
        return "", None, {}

    description = _multiline_input(
        "[bold]Description des règles[/bold] (terminer par une ligne 'END'):"
    )
    if not description.strip():
        console.print("[red]Description requise.[/red]")
        return name, None, {}

    param_space = _collect_parameter_space()

    console.print("\n[cyan]Génération de l'adaptateur via Claude…[/cyan]")
    try:
        from llm.adapter_generator import AdapterGenerator

        generator = AdapterGenerator()
        path = generator.generate(name, description)
    except Exception as e:
        console.print(f"[red]Échec de la génération : {type(e).__name__}: {e}[/red]")
        return name, None, param_space

    console.print(f"[green]Adapter écrit : {path}[/green]")
    code = path.read_text(encoding="utf-8")
    console.print(Syntax(code, "python", line_numbers=True, theme="monokai"))

    adapter_class = _import_adapter_class(name)
    if adapter_class is None:
        return name, None, param_space

    if Confirm.ask("Valider via 10 parties random-vs-random ?", default=True):
        report = generator.validate_adapter(adapter_class)
        _print_validation_report(report)
        if not report["ok"] and not Confirm.ask(
            "Validation échouée. Utiliser quand même ?", default=False
        ):
            console.print(
                f"[yellow]Édite {path} à la main puis relance.[/yellow]"
            )
            return name, None, param_space

    return name, adapter_class, param_space


def _collect_parameter_space() -> dict[str, tuple]:
    console.print(
        "\n[bold]Paramètres variables pour l'équilibrage[/bold] "
        "[dim](nom vide pour terminer)[/dim]"
    )
    space: dict[str, tuple] = {}
    while True:
        pname = Prompt.ask("Nom du paramètre", default="").strip()
        if not pname:
            break
        ptype = Prompt.ask("  Type", choices=["int", "float"], default="int")
        if ptype == "int":
            lo = IntPrompt.ask("  Min")
            hi = IntPrompt.ask("  Max")
        else:
            lo = FloatPrompt.ask("  Min")
            hi = FloatPrompt.ask("  Max")
        space[pname] = (ptype, lo, hi)
    return space


# -------------------------------------------------------------------- simulation


def _configure_simulation() -> dict[str, Any]:
    console.rule("[bold]Configuration de la simulation[/bold]")
    n_games = IntPrompt.ask("Nombre de parties", default=100)
    agent_type = Prompt.ask(
        "Type d'agent",
        choices=["random", "greedy", "mcts", "selfplay"],
        default="mcts",
    )
    mcts_sims = 100
    if agent_type in ("mcts", "selfplay"):
        mcts_sims = IntPrompt.ask("MCTS simulations par coup", default=100)
    optimize = Confirm.ask("Activer l'optimisation des paramètres ?", default=False)
    return {
        "n_games": n_games,
        "agent_type": agent_type,
        "mcts_simulations": mcts_sims,
        "optimize": optimize,
    }


def _build_agents(
    adapter: BaseAdapter,
    game_name: str,
    agent_type: str,
    mcts_sims: int,
) -> list:
    n_players = adapter.get_n_players()
    if agent_type == "random":
        return [RandomAgent(seed=i) for i in range(n_players)]
    if agent_type == "greedy":
        eval_fn = _try_load_eval(game_name)
        if eval_fn is None:
            console.print(
                "[yellow]Aucune fonction d'évaluation trouvée ; greedy se comportera "
                "comme random.[/yellow]"
            )
        return [GreedyAgent(adapter, eval_fn=eval_fn, seed=i) for i in range(n_players)]
    if agent_type == "mcts":
        return [
            MCTSAgent(adapter, n_simulations=mcts_sims, seed=i)
            for i in range(n_players)
        ]
    if agent_type == "selfplay":
        trained = _train_self_play(adapter, mcts_sims)
        return [trained for _ in range(n_players)]
    raise ValueError(f"Type d'agent inconnu : {agent_type}")


def _train_self_play(adapter: BaseAdapter, mcts_sims: int):
    console.rule("[bold]Pré-entraînement self-play[/bold]")
    n_iter = IntPrompt.ask("Itérations", default=3)
    n_games_per_iter = IntPrompt.ask("Parties par itération", default=20)
    eval_games = IntPrompt.ask("Parties d'évaluation", default=10)
    trainer = SelfPlayTrainer(
        adapter,
        n_iterations=n_iter,
        n_games_per_iter=n_games_per_iter,
        eval_games=eval_games,
        mcts_simulations=mcts_sims,
        verbose=True,
    )
    agent, _history = trainer.train()
    console.print("[green]Entraînement terminé.[/green]")
    return agent


def _run_simulation(
    adapter: BaseAdapter,
    game_name: str,
    config: dict[str, Any],
) -> list[GameResult]:
    console.rule(f"[bold]Simulation : {config['n_games']} parties[/bold]")
    agents = _build_agents(
        adapter, game_name, config["agent_type"], config["mcts_simulations"]
    )
    engine = SimulationEngine(adapter, agents, record=True, max_turns=1000)
    return [
        engine.run_game()
        for _ in track(
            range(config["n_games"]),
            description="Playing games",
            console=console,
        )
    ]


# -------------------------------------------------------------------- optimization


def _run_optimization(
    adapter_class: type[BaseAdapter],
    param_space: dict[str, tuple],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    console.rule("[bold]Optimisation Optuna[/bold]")
    n_trials = IntPrompt.ask("Nombre de trials", default=20)
    n_games_per_trial = IntPrompt.ask("Parties par trial", default=15)
    balance_targets = {
        "win_rate_range": (0.45, 0.55),
        "avg_turns_range": (3, 200),
    }
    console.print(
        f"[dim]Cibles : win_rate in {balance_targets['win_rate_range']}, "
        f"avg_turns in {balance_targets['avg_turns_range']}[/dim]"
    )

    def _factory(params: dict[str, Any]) -> BaseAdapter:
        return adapter_class(**params)

    optimizer = BalanceOptimizer(
        adapter_factory=_factory,
        param_space=param_space,
        balance_targets=balance_targets,
        n_trials=n_trials,
        n_games_per_trial=n_games_per_trial,
        mcts_simulations=max(20, config["mcts_simulations"] // 2),
        seed=0,
    )
    try:
        best = optimizer.optimize()
    except TypeError as e:
        console.print(
            f"[red]Le constructeur de {adapter_class.__name__} n'accepte pas "
            f"ces paramètres : {e}[/red]"
        )
        return None
    optimizer.print_best_params()
    return best


# -------------------------------------------------------------------- reporting


def _print_report(
    results: list[GameResult],
    best_params: dict[str, Any] | None,
    game_name: str,
) -> None:
    stats = StatsCollector(results)
    summary = stats.summary()

    console.rule(f"[bold]Rapport : {game_name}[/bold]")

    general = Table(title="Stats générales", show_header=True, header_style="bold cyan")
    general.add_column("Métrique", style="cyan")
    general.add_column("Valeur", justify="right")
    general.add_row("Parties jouées", str(summary["n_games"]))
    general.add_row("Tours moyens", f"{summary['avg_turns']:.1f}")
    general.add_row("Durée moyenne (ms)", f"{summary['avg_duration_ms']:.1f}")
    general.add_row("Timeouts", str(summary["timed_out_games"]))
    console.print(general)

    players = Table(title="Par joueur", show_header=True, header_style="bold cyan")
    players.add_column("Joueur", style="cyan", justify="center")
    players.add_column("Win rate", justify="right")
    players.add_column("Score (mean +/- std)", justify="right")
    players.add_column("Illegal/tour", justify="right")
    for pid in sorted(summary["win_rates"]):
        sd = summary["score_distribution"][pid]
        players.add_row(
            str(pid),
            f"{summary['win_rates'][pid]:.1%}",
            f"{sd['mean']:.2f} ± {sd['stdev']:.2f}",
            f"{summary['illegal_action_rates'][pid]:.4f}",
        )
    console.print(players)

    analyzer = DominanceAnalyzer(results)
    issues = analyzer.report()
    if issues:
        console.print("\n[bold]Pathologies d'équilibrage[/bold]")
        for issue in issues:
            color = _SEVERITY_COLORS.get(issue.severity, "white")
            console.print(
                f"  [{color}][{issue.severity.upper():>8}][/{color}] "
                f"{issue.category}: {issue.description}"
            )
    else:
        console.print("\n[green]Aucune pathologie détectée.[/green]")

    if best_params is not None:
        console.print("\n[bold]Paramètres optimaux[/bold]")
        for k, v in best_params.items():
            console.print(f"  {k} = {v}")


# -------------------------------------------------------------------- helpers


def _instantiate_safe(adapter_class: type[BaseAdapter]) -> BaseAdapter | None:
    try:
        return adapter_class()
    except TypeError as e:
        console.print(
            f"[red]{adapter_class.__name__}() requires arguments and the CLI "
            f"can't supply them: {e}[/red]"
        )
        return None


def _import_adapter_class(name: str) -> type[BaseAdapter] | None:
    module_path = f"games.{name}.adapter"
    try:
        # Force a fresh import so a freshly-generated adapter is picked up.
        import sys as _sys

        _sys.modules.pop(module_path, None)
        module = importlib.import_module(module_path)
    except Exception as e:
        console.print(f"[red]Impossible d'importer {module_path} : {e}[/red]")
        return None
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseAdapter)
            and obj is not BaseAdapter
        ):
            return obj
    console.print(f"[red]Aucune sous-classe de BaseAdapter trouvée dans {module_path}.[/red]")
    return None


def _try_load_eval(game_name: str) -> Callable | None:
    module_path = f"games.{game_name}.eval"
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return None
    expected = f"{game_name}_eval"
    if hasattr(module, expected):
        return getattr(module, expected)
    # Fallback: the first public callable that isn't a factory or class.
    for attr in dir(module):
        if attr.startswith("_") or attr.startswith("make_"):
            continue
        obj = getattr(module, attr)
        if callable(obj) and not isinstance(obj, type):
            return obj
    return None


def _multiline_input(prompt: str) -> str:
    console.print(prompt)
    console.print("[dim](tape une ligne 'END' pour terminer)[/dim]")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def _print_validation_report(report: dict[str, Any]) -> None:
    status_color = "green" if report["ok"] else "red"
    console.print(
        Panel(
            f"Instantiation : {report['instantiation']}\n"
            f"BaseAdapter   : {report['is_base_adapter']}\n"
            f"N players     : {report['n_players']}\n"
            f"Games OK      : {report['games_completed']} / {report['games_attempted']}\n"
            + (
                "Exceptions    :\n  - " + "\n  - ".join(report["exceptions"])
                if report["exceptions"]
                else "Exceptions    : aucune"
            ),
            title="Validation de l'adapter",
            border_style=status_color,
        )
    )


if __name__ == "__main__":
    main()
