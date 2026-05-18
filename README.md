# shako

Game-balancing framework. Define a game once via a small interface, then run
self-play, MCTS rollouts, dominance analysis, and Optuna-driven parameter
sweeps against it — no per-game scaffolding. New games can also be bootstrapped
from a plain-English description via the Claude API.

## Installation

Requires Python 3.11+ (older versions work thanks to `from __future__ import
annotations`, but the project is pinned to 3.11 in `pyproject.toml`).

```bash
git clone <your-repo-url> shako
cd shako
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .                                    # makes `core`, `rl`, `games`, ... importable
```

For LLM-generated adapters, also set:

```bash
export ANTHROPIC_API_KEY=sk-ant-...                 # set ANTHROPIC_API_KEY=... on Windows cmd
```

## Quickstart

Run 50 MCTS-vs-MCTS games of Nim and print balance stats:

```python
from games.nim.adapter import NimAdapter
from core.engine import SimulationEngine
from core.stats import StatsCollector
from rl.mcts_agent import MCTSAgent

adapter = NimAdapter(n_sticks=21)
agents = [MCTSAgent(adapter, n_simulations=100, seed=i) for i in range(2)]
engine = SimulationEngine(adapter, agents, record=True)
results = [engine.run_game() for _ in range(50)]

StatsCollector(results).print_report()
```

Or use the interactive CLI:

```bash
python -m cli       # prompts for game, agent type, simulation config, then reports
```

## Describing a new game

Two paths.

**LLM-generated.** Provide a free-text description and let Claude write the
adapter. Use the CLI's "new" flow, or call the generator directly:

```python
from llm.adapter_generator import AdapterGenerator

description = """
Two-player Tic-Tac-Toe on a 3x3 grid. Players alternate placing X / O.
First to align three marks wins (+1). Full board with no winner is a draw (0,0).
Player 0 moves first.
"""

gen = AdapterGenerator()
path = gen.generate("tic_tac_toe", description)          # writes games/tic_tac_toe/adapter.py
report = gen.validate_adapter(_load(path))               # runs 10 random games, reports problems
```

**Hand-written.** Subclass [`BaseAdapter`](core/base_adapter.py) and implement
every abstract method. [`games/nim/adapter.py`](games/nim/adapter.py) is the
canonical reference; [`games/cards/adapter.py`](games/cards/adapter.py) shows
the hidden-information pattern with a `sample_state` method for MCTS
determinization.

## Architecture

```
shako/
├── core/        Game-agnostic interfaces, engine, stats
│   ├── base_adapter.py     BaseAdapter ABC — every game implements this
│   ├── base_agent.py       BaseAgent ABC
│   ├── types.py            State, ObservableState, Action, GameResult
│   ├── engine.py           SimulationEngine — turn loop + multiprocessing batch
│   └── stats.py            StatsCollector — win rates, score distributions
│
├── games/       Concrete adapters
│   ├── nim/                perfect-information reference
│   └── cards/              hidden-information reference (with sample_state)
│
├── rl/          Agents
│   ├── random_agent.py     baseline / fallback
│   ├── greedy_agent.py     1-step lookahead with pluggable eval_fn
│   ├── mcts_agent.py       UCT + per-simulation determinization
│   ├── human_agent.py      console-driven player
│   └── self_play.py        SelfPlayTrainer + PolicyMCTSAgent
│
├── balancer/    Equilibrium tooling
│   ├── optimizer.py        BalanceOptimizer (Optuna TPE search)
│   └── analyzer.py         DominanceAnalyzer (seat advantage, dead actions, …)
│
├── llm/         Claude-driven code generation
│   ├── adapter_generator.py   description → games/<name>/adapter.py
│   └── eval_generator.py      criteria → games/<name>/eval.py
│
├── cli/         Interactive interface (python -m cli)
└── tests/       pytest suite
```

**Data flow.** Adapter implements game rules → Agent picks actions from the
ObservableState the adapter exposes → Engine runs Agent-vs-Agent through the
Adapter, producing `GameResult[]` → StatsCollector and DominanceAnalyzer
aggregate that into metrics and balance pathologies → BalanceOptimizer wraps
the entire pipeline in an Optuna search over adapter constructor parameters.

`BaseAdapter` is the single integration point: anything that implements its
nine methods works with every agent, the engine, the analyzer, and the
optimizer without modification.

## Running tests

```bash
pytest tests/                       # full suite
pytest tests/test_nim.py -v         # one module
ANTHROPIC_API_KEY=... pytest tests/test_generators.py  # live LLM test (skipped without the key)
```
