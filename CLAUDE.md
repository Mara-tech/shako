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
| `ui/rich_agent.py` | `RichHumanAgent` — Rich-styled human player (coloured board, styled actions) |
| `ui/textual_agent.py` | `TextualHumanAgent` — bridge between engine thread and Textual event loop |
| `ui/textual_app.py` | `ShakTUIApp` — Textual TUI application (board panel, action list, replay) |
| `ui/grid_widget.py` | `GridWidget` — clickable NxM grid widget (column mode for Connect Four, cell mode for Tic-Tac-Toe) |

## Adding a new game

Subclass `BaseAdapter` and implement its 9 abstract methods. Reference implementations:

- **Perfect information:** `games/nim/adapter.py` (canonical, simplest)
- **Hidden information:** `games/cards/adapter.py` (uses `sample_state` for MCTS determinization)
- **Grid-based UI:** `games/tictactoe/adapter.py` (cell mode) or `games/connect4/adapter.py` (column mode) — `get_grid_config`/`get_grid_render_config`/`get_action_for_click` wiring for the Textual clickable-grid widget

Override `get_action_label(action)` in adapters with large combinatorial action spaces
to return a coarse category string — prevents spurious rare-action warnings.

Trained self-play policies are saved under `games/<name>/models/selfplay/<params>/`.

## Key invariant

Anything that correctly implements `BaseAdapter` works with every agent, the engine,
the analyzer, and the optimizer without modification.

## Rules

- Project language is English. The only exception can be the name of some games that may not exist in English, so it would be considered as a proper noun in the provided language. Moreover, if a game name is provided in another language than English and is well-known enough to have an English name, use it (e.g. when you are prompted to implement "Morpion" (FR), consider you have to implement "TicTacToe").
- **Never modify `BaseAdapter`'s interface** (the 9 abstract method signatures) without discussing it first — it is the single integration point; everything else depends on it.
- **Run `pytest tests/` before marking a task done.** If tests fail, fix them before concluding.
- **Pick the right reference implementation** for a new game: `nim/` by default, `cards/` for hidden information, `tictactoe/` or `connect4/` for a clickable grid UI.
- **Create a `rules.md`** in English in the game's folder when adding a new game. Describe the rules in plain language: players, actions, win/draw/loss conditions, edge cases.
- **`get_scores()` must give a draw a value strictly between a win and a loss** (e.g. 0.5 when a win is 1.0 and a loss is 0.0), never the same value as a loss. MCTS backprop (`rl/mcts_agent.py`) uses these raw scores as its reward signal, so a draw scored identically to a loss makes the search permanently indifferent between "secure the draw" and "lose" — more simulations won't fix it, since the tie is exact, not noise. See `games/connect4/adapter.py`'s `get_scores` for the reference pattern.
- **Don't bake a multi-round match (N repeated games + a running scoreboard) into an adapter's own state.** `is_terminal` should end at the atomic game, one call to `get_initial_state` per game. Running several games back-to-back and tallying results is a caller concern — the CLI's replay loop, `SelfPlayTrainer`, `DominanceAnalyzer`, `StatsCollector` — all of which already aggregate independent `GameResult`s generically. Folding rounds into one session dilutes MCTS's per-move reward signal across unrelated future rounds and hides seat/starter advantage from the balance-analysis tools, which only see one aggregated result per session instead of one per game.
- **`get_initial_state()` should fix a deterministic starting player (player 0), not parameterize it.** `nim`, `cards`, and `connect4` all do this. Whoever effectively "goes first" against a given opponent is a seat-assignment concern of the caller (e.g. `SelfPlayTrainer._evaluate` already alternates which agent occupies which seat) — an adapter-level `starting_player` option is redundant with that and adds a second, easy-to-desync source of truth.
- **Create a test** when adding a new game in `tests/` folder.
- **Update the Module map table in this file** when a new top-level module is added to the project.
