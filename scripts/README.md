# scripts/train.py — Non-interactive self-play trainer

Runs a full self-play training loop for any game adapter and saves the resulting
agent to disk. Designed for CI pipelines, overnight runs, or any context where
you don't want the interactive CLI.

```bash
python scripts/train.py --game nim --n-iterations 20 --seed 42
python scripts/train.py --game connect4 --adapter-rows 6 --adapter-cols 7 --n-iterations 30 --mcts-simulations 500
```

## Flags

| Flag | Default | Description |
|---|---|---|
| `--game` | *(required)* | Game folder name under `games/` |
| `--n-iterations` | `10` | Number of train/eval cycles |
| `--n-games-per-iter` | `50` | Self-play games per training iteration |
| `--eval-games` | `40` | Evaluation games used to decide promotion |
| `--mcts-simulations` | `100` | MCTS simulations per move |
| `--promotion-threshold` | `0.55` | Win-rate required to promote the candidate |
| `--seed` | *(random)* | RNG seed for reproducibility |
| `--adapter-<param>` | varies | Adapter constructor parameters (see below) |
| `--jordan-server-url` | *(disabled)* | Jordan server URL for remote monitoring |

### Adapter parameters

Each game adapter exposes its own constructor parameters as `--adapter-<name>`
flags. For example, `NimAdapter(n_sticks, max_take, last_takes_wins)` becomes:

```bash
python scripts/train.py --game nim \
    --adapter-n-sticks 21 \
    --adapter-max-take 3 \
    --adapter-last-takes-wins false
```

Run `--help` to see the full list for a given game:

```bash
python scripts/train.py --game connect4 --help
```

### Output

On completion the script saves the trained agent to:

```
games/<name>/models/selfplay/<adapter-params>/<timestamp>.pkl
```

The agent can be reloaded in the interactive CLI or via `SelfPlayTrainer.load_agent()`.

---

## Remote monitoring and control with Jordan

`--jordan-server-url` connects the training run to a
[Jordan](https://pypi.org/project/jordan-py/) server, which gives you a live
GUI showing progress and lets you stop the run early from a browser or any
Jordan client.

### Setup

```bash
pip install jordan-py        # or: pip install "shako[monitor]"
```

Start a Jordan server (see the [Jordan documentation](https://github.com/Mara-tech/jordan))
and then pass its URL to the script:

```bash
python scripts/train.py --game nim --n-iterations 30 --seed 0 \
    --jordan-server-url http://localhost:5000/jordan/
```

### What gets reported

| Event | Jordan call |
|---|---|
| Training started | `send_status` with game name, iteration count, MCTS sims, seed |
| After each iteration | `send_progress(0–100)` + `send_status` with win rate and promotion result |
| Training complete | `send_success_status` with promotion count and saved model path |
| Save failed | `send_failure_status` with the error message |

### Stopping early from the server

The client registers a single action named **`break_training_loop`**. Trigger it
from the Jordan server UI at any point during training. The script polls
`read_message()` after each iteration; when a `break_training_loop` message
arrives, it is acknowledged and marked processed, and training stops cleanly
after the current iteration — the best agent so far is saved normally.

This is useful when a run has clearly converged (win rate plateauing) and you
want to reclaim the machine without waiting for the remaining iterations.

### Without jordan-py installed

If you pass `--jordan-server-url` but `jordan-py` is not installed, the script
prints one line and continues training normally with stdout output only.
