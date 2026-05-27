# shako — Claude context

Game-balancing framework. Define a game via `BaseAdapter`, then run self-play,
MCTS, dominance analysis, and Optuna parameter sweeps against it for free.
Games can also be bootstrapped from plain-English descriptions via the Claude API.

## Commands

```bash
python -m cli                          # interactive CLI (select game, agents, run)
pytest tests/                          # full test suite
pytest tests/test_nim.py -v            # single module
ANTHROPIC_API_KEY=... pytest tests/test_generators.py  # live LLM test
```

Python env: `pip install -e .` from root. PYTHONPATH = project root.

## Module map

| Path | Role |
|---|---|
| `core/base_adapter.py` | `BaseAdapter` ABC — the single integration point |
| `core/base_agent.py` | `BaseAgent` ABC |
| `core/types.py` | `State`, `ObservableState`, `Action`, `GameResult` |
| `core/engine.py` | `SimulationEngine` — turn loop + multiprocessing batch |
| `core/stats.py` | `StatsCollector` — win rates, score distributions |
| `rl/mcts_agent.py` | UCT + per-simulation determinization |
| `rl/self_play.py` | `SelfPlayTrainer` + `PolicyMCTSAgent` (saved to `games/<name>/models/`) |
| `rl/greedy_agent.py` | 1-step lookahead with pluggable `eval_fn` |
| `balancer/analyzer.py` | `DominanceAnalyzer` — seat advantage, entropy, rare actions, runaway duration |
| `balancer/optimizer.py` | `BalanceOptimizer` — Optuna TPE search over adapter parameters |
| `llm/adapter_generator.py` | description → `games/<name>/adapter.py` via Claude API |
| `llm/eval_generator.py` | criteria → `games/<name>/eval.py` via Claude API |
| `cli/main.py` | interactive game selection and simulation |

## Adding a new game

Subclass `BaseAdapter` and implement its 9 abstract methods. Reference implementations:

- **Perfect information:** `games/nim/adapter.py` (canonical, simplest)
- **Hidden information:** `games/cards/adapter.py` (uses `sample_state` for MCTS determinization)
- **Multi-round / configurable start:** `games/tictactoe/adapter.py`

Override `get_action_label(action)` in adapters with large combinatorial action spaces
to return a coarse category string — prevents spurious rare-action warnings.

Trained self-play policies are saved under `games/<name>/models/selfplay/<params>/`.

## Key invariant

Anything that correctly implements `BaseAdapter` works with every agent, the engine,
the analyzer, and the optimizer without modification.

## Rules

- **Never modify `BaseAdapter`'s interface** (the 9 abstract method signatures) without discussing it first — it is the single integration point; everything else depends on it.
- **Run `pytest tests/` before marking a task done.** If tests fail, fix them before concluding.
- **Pick the right reference implementation** for a new game: `nim/` by default, `cards/` for hidden information, `tictactoe/` for multi-round or configurable starting player.
- **Create a `rules.md`** in the game's folder when adding a new game. Describe the rules in plain language: players, actions, win/draw/loss conditions, edge cases.
- **Create a test** when adding a new game in `tests/` folder.
- **Update the Module map table in this file** when a new top-level module is added to the project.
