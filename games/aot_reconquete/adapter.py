"""Shako adapter for AOT Reconquête, whose rules run in TypeScript behind a stdio bridge.

A 1-player, perfect-information game: the Scouts are the only side that chooses, the titans
are the game rather than an opponent. See `rules.md` next to this file.

## Where the rules live, and where this file stops
Not one rule is written here. Legality, effect, and the end of the game are answered by the
game's own controllers, reached through the Node process described in `bridge.py`. This file
names shako's nine methods over that protocol and nothing else — the arbitration the game's
repository made when it shipped the bridge: its side stays close to its domain, the
adaptation to `BaseAdapter` is Python's, where it costs less to correct when that base class
moves.

## The state is the whole game, and it travels
`State.data` is what the bridge calls an *observable state*: `{"game": <the save file, as
the app's "Save game" writes it>, "rng": <the generator's state>}`. Both halves are carried
verbatim; nothing here reshapes them. The dice ride beside the save rather than inside it
because a save file's format is a player's file and must not move for a search's benefit —
that is the game repository's arbitration, and the reason `apply_action` is a function of its
arguments at all.

The bridge itself holds states under opaque handles, to keep a large save off the pipe. This
adapter cannot use that, and deliberately so: `BaseAdapter` is stateless by contract, and
`core/engine.py` pickles the adapter into `spawn`ed workers. So every call uploads its state,
asks its question, and releases the handle — and the price of that is small enough to say out
loud. Measured on this machine on 2026-09-13, a 9x12 board and an 8 kB state, medians over 30
warm applies: **4.53 ms** for the upload/apply/download/release round trip against **3.62 ms**
for the bridge's own handle-to-handle apply. Statelessness costs ~0.9 ms, a quarter of the
call; the mount the game does inside it is most of the rest.

## Three consequences that are the point rather than the price
* **`clone_state` never touches the pipe.** It is a `copy.deepcopy` of a plain JSON dict, so
  `rl/mcts_agent.py`, `rl/self_play.py` and `rl/greedy_agent.py` can call it in a loop.
* **The adapter survives pickling.** `__getstate__` drops the process and its lock; each
  worker relaunches its own on first use. Without it `run_batch(n_workers=2)` never starts —
  `TypeError: cannot pickle '_thread.RLock' object`, and the `Popen` behind it.
* **`get_legal_actions` is never empty on a non-terminal state**, which `BaseAdapter`'s
  docstring requires. The bridge already honours it — out of combat the end of turn is always
  offered, in combat the capture is — so this file checks and *reports*, rather than
  fabricating an action to hide a bridge that stopped honouring it.
"""

from __future__ import annotations

import copy
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Self

from core.base_adapter import BaseAdapter
from core.types import Action, ObservableState, State
from games.aot_reconquete.bridge import (
    AotReconqueteError,
    BridgeProtocolError,
    ShakoBridge,
)

# The one player, the Scouts. `describe` answers the same two numbers, and the handshake
# checks it: a game that grew a second seat must not be played as if it had one.
_PLAYER_COUNT = 1
_ONLY_PLAYER = 0

# Written afresh by the game on every serialisation, so it differs between two states that
# describe the same position. Excluded from the comparison `apply_action` makes below.
_VOLATILE_SAVE_FIELDS = frozenset({"savedAt"})


class IllegalActionError(AotReconqueteError):
    """An action the game refused to play.

    The protocol does not report one: a well-formed action that this position does not allow
    is refused by the game's own guards, exactly as it would be under a player's mouse, and
    the bridge answers with the position unchanged. Left unsaid, that is a search silently
    looping on a move that never happens, so `apply_action` says it instead.
    """


class AotReconqueteAdapter(BaseAdapter):
    """AOT Reconquête, relayed to the TypeScript game over a JSON stdio pipe.

    Requires a checkout of the game repository with its dependencies installed
    (`npm ci`). It is looked for in `game_dir`, then `$AOT_RECONQUETE_DIR`, then as a
    sibling of the shako checkout.
    """

    def __init__(self, game_dir: str | None = None, seed: int | None = None) -> None:
        """Args:
            game_dir: the `aot-reconquete.js` checkout. `None` resolves it as described above.
            seed: pins the game `get_initial_state` builds. `None` (default) lets the game
                roll its own scouts, so every game of a batch differs — which is what a
                balance study wants. A seed makes every game of that batch *the same game*;
                use it to reproduce one, not to run a hundred.
        """
        self.game_dir = game_dir
        self.seed = seed
        self._bridge: ShakoBridge | None = None
        # Guards the lazy launch below. `core/engine.py` runs an agent in a worker thread when
        # `max_action_ms` is set, and a timed-out agent keeps running: two threads reaching a
        # fresh adapter at once would each start a process, and one of them would be orphaned.
        self._lock = threading.RLock()

    # ------------------------------------------------------- the nine methods

    def get_initial_state(self) -> State:
        params: dict[str, Any] = {} if self.seed is None else {"seed": self.seed}
        handle = self._call("get_initial_state", **params)["state"]
        try:
            return self._download(handle)
        finally:
            self._release(handle)

    def get_n_players(self) -> int:
        return _PLAYER_COUNT

    def get_current_player(self, state: State) -> int:
        return _ONLY_PLAYER

    def is_terminal(self, state: State) -> bool:
        with self._held(state) as handle:
            return bool(self._call("is_terminal", state=handle)["terminal"])

    def get_legal_actions(self, state: State, player_id: int) -> list[Action]:
        """Every action the Scouts may play, for every force on the board.

        The emptiness check is the invariant `BaseAdapter` states, verified where it can
        still be attributed: an empty list on a live position is a bug in the game's
        enumeration — the end of turn is always legal out of combat — and this raises rather
        than inventing the action that is missing. Terminality is asked on the same handle,
        so the check costs nothing until the list is actually empty.
        """
        with self._held(state) as handle:
            raw = self._call("get_legal_actions", state=handle)["actions"]
            if not raw and not self._call("is_terminal", state=handle)["terminal"]:
                raise BridgeProtocolError(
                    "the bridge offered no legal action on a non-terminal state. Out of "
                    "combat the end of turn is always legal and in combat the capture is, "
                    "so this is a bug to report to the game's repository "
                    "(BoardController.getLegalActions)."
                )
        return [Action(data=action) for action in raw]

    def apply_action(self, state: State, action: Action, player_id: int) -> State:
        """The position that follows, as a fresh `State`. `state` is never touched.

        Raises `IllegalActionError` when the game refused the action. See `_changed` for what
        "refused" is read from, and why it cannot be a plain equality.
        """
        with self._held(state) as handle:
            applied = self._call("apply_action", state=handle, action=action.data)["state"]
            try:
                result = self._download(applied)
            finally:
                self._release(applied)
        if not _changed(state.data, result.data):
            raise IllegalActionError(
                f"the game refused {self.get_action_display(action)}: it left the position "
                f"unchanged. Choose from get_legal_actions()."
            )
        return result

    def get_scores(self, state: State) -> dict[int, float]:
        """0 on defeat and while the game is on, `1 / turns` on victory.

        The game's own scale, read off the bridge rather than recomputed: a game won in ten
        turns scores ten times one won in a hundred, so a search does not dawdle.
        """
        with self._held(state) as handle:
            scores = self._call("get_scores", state=handle)["scores"]
        if not isinstance(scores, list) or len(scores) != _PLAYER_COUNT:
            raise BridgeProtocolError(f"get_scores: expected one score per player, got {scores!r}")
        return {_ONLY_PLAYER: float(scores[0])}

    def get_observable_state(self, state: State, player_id: int) -> ObservableState:
        """The identity: one player, and nothing on this board is hidden from them.

        The bridge keeps a method for it all the same, so the day the game grows a fog of war
        there is one place for it to land.
        """
        return ObservableState(data=dict(state.data), player_id=player_id)

    def clone_state(self, state: State) -> State:
        """A deep copy in Python, which never reaches the pipe.

        `State.data` is plain JSON — no cycle, no object — so `deepcopy` is a true copy, and
        MCTS can clone per simulation without a round trip.
        """
        return State(data=copy.deepcopy(state.data))

    # --------------------------------------------------------- display helpers

    def get_action_label(self, action: Action) -> str:
        """The action's kind alone — `move`, `attack`, `end_of_turn`…

        `balancer/analyzer.py` counts these to flag rare actions. The full action carries the
        tiles it acts on, so every move on a 9x12 board would be its own category and every
        one of them would look rare. The kind is the category that means something.
        """
        kind = action.data.get("kind") if isinstance(action.data, dict) else None
        return str(kind) if kind is not None else json.dumps(action.data, sort_keys=True)

    def get_action_display(self, action: Action) -> str:
        """One line for a human picking an action in the CLI or the TUI."""
        data = action.data if isinstance(action.data, dict) else {}
        kind = str(data.get("kind", "?"))
        if "from" in data and "to" in data:
            return f"{kind} {_tile(data['from'])} -> {_tile(data['to'])}"
        if "on" in data:
            scout = f" ({data['scout']})" if "scout" in data else ""
            return f"{kind} on {_tile(data['on'])}{scout}"
        if "bodyPart" in data:
            return f"{kind} {data['bodyPart']}"
        return kind

    def get_rich_renderable(self, obs_state: ObservableState):
        """A short status panel: the turn, and what the win and loss conditions stand at."""
        from rich.text import Text

        game = obs_state.data.get("game", {}) if isinstance(obs_state.data, dict) else {}
        terrain = game.get("terrain", [])
        kinds = [tile.get("kind") for tile in terrain if isinstance(tile, dict)]
        text = Text()
        text.append(f"Turn {game.get('turnCounter', '?')}\n\n", style="bold")
        text.append(f"Functional gates: {kinds.count('door')} (4 to win)\n", style="green")
        text.append(f"Breached gates:   {kinds.count('broken_door')}\n", style="red")
        text.append(f"Scout forces:     {len(game.get('scoutForces', []))}\n")
        text.append(f"Titan swarms:     {len(game.get('titanSwarms', []))}\n")
        return text

    # ------------------------------------------------------------- the process

    def bridge(self) -> ShakoBridge:
        """The Node process, launched on first use.

        Lazy because of pickling: a worker receives an adapter with no process and starts its
        own here, on the first question it asks.
        """
        with self._lock:
            if self._bridge is None:
                started = ShakoBridge(self.game_dir)
                started.start()
                self._bridge = started
                if (started.players, started.current_player) != (_PLAYER_COUNT, _ONLY_PLAYER):
                    seats = (started.players, started.current_player)
                    self.close()
                    raise BridgeProtocolError(
                        f"the game now describes (players, currentPlayer) = {seats}; this "
                        f"adapter is written for ({_PLAYER_COUNT}, {_ONLY_PLAYER}) and would "
                        f"play the wrong seats."
                    )
            return self._bridge

    def close(self) -> None:
        """Stop the Node process. Idempotent; the adapter restarts one if asked again."""
        with self._lock:
            if self._bridge is not None:
                self._bridge.close()
                self._bridge = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        """Pickle everything but the process — `core/engine.py` requires it.

        `run_batch(n_workers > 1)` pickles the adapter into `spawn`ed workers, and a
        `subprocess.Popen` cannot be pickled. Dropping it here is what makes the batch run;
        `bridge()` relaunches one per worker, on demand.
        """
        state = self.__dict__.copy()
        state["_bridge"] = None
        state.pop("_lock", None)  # a lock cannot be pickled either, and means nothing elsewhere
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._bridge = None
        self._lock = threading.RLock()

    # --------------------------------------------------------------- internals

    def _call(self, method: str, **params: Any) -> dict[str, Any]:
        return self.bridge().call(method, **params)

    @contextmanager
    def _held(self, state: State) -> Iterator[str]:
        """Upload `state`, yield its handle, and free it whatever happens.

        The bridge frees nothing on its own — `release_state` is the only thing that does — so
        a search that skipped this would keep every position it ever visited.
        """
        handle = self._upload(state)
        try:
            yield handle
        finally:
            self._release(handle)

    def _upload(self, state: State) -> str:
        """Hand a state to the bridge and get the handle every other method names it by."""
        data = state.data
        if not isinstance(data, dict):
            raise AotReconqueteError(
                f"a State of this game carries a dict, got {type(data).__name__}"
            )
        if "game" not in data or "rng" not in data:
            raise AotReconqueteError(
                "a State of this game carries {'game': <save file>, 'rng': <int>}, as "
                f"get_initial_state() builds it; got keys {sorted(data)!r}"
            )
        return self._call("from_save", game=data["game"], rng=data["rng"])["state"]

    def _download(self, handle: str) -> State:
        return State(data=self._call("get_observable_state", state=handle)["observed"])

    def _release(self, handle: str) -> None:
        self._call("release_state", state=handle)


def _tile(coordinates: Any) -> str:
    if isinstance(coordinates, dict):
        return f"({coordinates.get('col')},{coordinates.get('row')})"
    return str(coordinates)


def _changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Did playing the action move the game at all?

    Two states that describe the same position are **not** equal as dicts, so this cannot be
    `before != after`. Measured on 2026-09-13, replaying an action the game refuses:

    * `game.savedAt` is stamped afresh on every serialisation, so it always differs;
    * the `terrain` array comes back in another order, the game rebuilding it from a map.

    So the comparison drops the volatile fields and orders every array. The two cheap fields
    are tried first — the generator's state and the turn counter — because an action that did
    anything nearly always moves one of them, which keeps the full canonicalisation off the
    common path.

    The direction that matters is the one this is used for: a *legal* action must never look
    unchanged. Verified over 105 legal actions of six kinds across three seeded games — none
    did. A refused one did, every time.
    """
    if before.get("rng") != after.get("rng"):
        return True
    if _turn_counter(before) != _turn_counter(after):
        return True
    return _position_of(before) != _position_of(after)


def _turn_counter(state: dict[str, Any]) -> Any:
    game = state.get("game")
    return game.get("turnCounter") if isinstance(game, dict) else None


def _position_of(state: Any) -> str:
    """A canonical string for a position: volatile fields dropped, every array ordered."""
    return json.dumps(_canonical(state), sort_keys=True, default=str)


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in value.items() if k not in _VOLATILE_SAVE_FIELDS}
    if isinstance(value, list):
        return sorted(
            (_canonical(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        )
    return value
