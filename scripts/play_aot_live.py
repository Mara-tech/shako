#!/usr/bin/env python
"""Play one game of AOT Reconquête with MCTS — watched live in the game's page, and recorded.

End to end, from two terminals (the game repository is `aot-reconquete.js`, cloned next to
shako or pointed at by `$AOT_RECONQUETE_DIR`):

    # once, in aot-reconquete.js
    npm ci

    # terminal 1, in aot-reconquete.js — the game's page, on http://localhost:8080
    npm run dev

    # terminal 2, in shako
    python scripts/play_aot_live.py --seed 168

    # then open, in a browser — the address is also printed by the script:
    http://localhost:8080/?spectate=8090

The page can be opened before the game starts, during it, or reloaded at any moment: the
channel sends a watcher the whole game as it connects, then each move as it is played.

Afterwards the game is in the file the script printed (by default under
`games/aot_reconquete/recordings/`). To replay it: on the same `npm run dev` page, main menu,
**Replay a Game** — shown by the dev server only — and pick that file. The file is rewritten
whole after every move, so it can also be opened while the game is still going.

What is shown and recorded is **the game played, and nothing of the search**: the MCTS
agent applies thousands of actions through the same bridge, and only the moves the engine
actually played are published (`games/aot_reconquete/publisher.py`).

One game, in this process. A batch across worker processes cannot be watched — each worker
would launch its own bridge on the same port and write the same file — and the adapter
refuses it; see `games/aot_reconquete/adapter.py`.

Flags:
    --port N          live channel port (default 8090); 0 lets the system pick one
    --no-live         no live channel
    --record PATH     recording file (default games/aot_reconquete/recordings/<time>.json)
    --no-record       no recording
    --seed N          pins the game (its board and scouts); default: a new game each run
    --time-ms N       MCTS thinking time per move (default 2000)
    --rollout-depth N MCTS random playout depth (default 10)
    --max-turns N     stop after this many moves (default 300)
    --page URL        where `npm run dev` serves the page (default http://localhost:8080)
    --game-dir PATH   the aot-reconquete.js checkout (default: $AOT_RECONQUETE_DIR, or a sibling)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core.engine import SimulationEngine  # noqa: E402
from core.types import Action, State  # noqa: E402
from games.aot_reconquete.adapter import AotReconqueteAdapter  # noqa: E402
from games.aot_reconquete.publisher import MirrorPublisher  # noqa: E402
from rl.mcts_agent import MCTSAgent  # noqa: E402

_DEFAULT_PORT = 8090  # the one `shako-bridge.md` uses in its examples
_DEFAULT_PAGE = "http://localhost:8080"  # where the game's `npm run dev` serves it
_RECORDINGS_DIR = _ROOT / "games" / "aot_reconquete" / "recordings"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play one AOT Reconquête game with MCTS, live in the game's page and recorded.",
        epilog="See this script's docstring for the end-to-end commands, `npm run dev` included.",
    )
    live = parser.add_mutually_exclusive_group()
    live.add_argument("--port", type=int, default=_DEFAULT_PORT,
                      help=f"live channel port (default {_DEFAULT_PORT}; 0 = any free port)")
    live.add_argument("--no-live", action="store_true", help="no live channel")
    record = parser.add_mutually_exclusive_group()
    record.add_argument("--record", type=Path, default=None,
                        help="recording file (default games/aot_reconquete/recordings/<time>.json)")
    record.add_argument("--no-record", action="store_true", help="no recording")
    parser.add_argument("--seed", type=int, default=None, help="pins the game; default: a new one")
    parser.add_argument("--time-ms", type=int, default=2000, help="MCTS time per move (ms)")
    parser.add_argument("--rollout-depth", type=int, default=10, help="MCTS playout depth")
    parser.add_argument("--max-turns", type=int, default=300, help="stop after this many moves")
    parser.add_argument("--page", default=_DEFAULT_PAGE, help="where `npm run dev` serves the page")
    parser.add_argument("--game-dir", default=None, help="the aot-reconquete.js checkout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    live_port = None if args.no_live else args.port
    record_file: Path | None = None
    if not args.no_record:
        record_file = args.record or _RECORDINGS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.json"
        record_file = record_file.expanduser().resolve()
        record_file.parent.mkdir(parents=True, exist_ok=True)

    adapter = AotReconqueteAdapter(
        game_dir=args.game_dir,
        seed=args.seed,
        live_port=live_port,
        record_file=str(record_file) if record_file else None,
    )
    with adapter, MirrorPublisher(adapter) as publisher:
        # Started now rather than on the first move, so the address is known — and a port
        # already taken reported — before anything is played.
        bridge = adapter.bridge()
        if bridge.bound_live_port is not None:
            print(f"Watch live:  {args.page.rstrip('/')}/?spectate={bridge.bound_live_port}")
            print("             (needs `npm run dev` in the game repository)")
        if record_file is not None:
            print(f"Recording:   {record_file}")
            print("             (replay: `npm run dev`, main menu, \"Replay a Game\")")
        print(f"MCTS: {args.time_ms} ms per move, playouts of {args.rollout_depth}, "
              f"at most {args.max_turns} moves\n")

        def publish_and_report(state: State, action: Action, player_id: int, next_state: State) -> None:
            publisher(state, action, player_id, next_state)
            turn = next_state.data["game"].get("turnCounter", "?")
            print(f"  {publisher.published_actions:4d}. turn {turn}: "
                  f"{adapter.get_action_display(action)}", flush=True)

        agent = MCTSAgent(
            adapter,
            time_limit_ms=args.time_ms,
            max_rollout_depth=args.rollout_depth,
            seed=args.seed,
        )
        engine = SimulationEngine(
            adapter, [agent], max_turns=args.max_turns, on_action_applied=publish_and_report
        )
        try:
            result = engine.run_game()
        except KeyboardInterrupt:
            print(f"\nInterrupted after {publisher.published_actions} moves.")
            if record_file is not None and publisher.published_actions:
                print(f"The recording holds them: {record_file}")
            return 130

    if result.timed_out:
        outcome = f"stopped at --max-turns {args.max_turns}, game not over"
    elif result.scores[0] > 0:
        outcome = f"victory, score {result.scores[0]:.4f} (1 / turns)"
    else:
        outcome = "defeat"
    print(f"\n{result.n_turns} moves in {result.duration_ms / 1000:.0f} s: {outcome}.")
    if record_file is not None:
        print(f"Recorded in {record_file}")
    return 0


if __name__ == "__main__":
    from core.stdio import make_stdio_printable

    make_stdio_printable()
    raise SystemExit(main())
