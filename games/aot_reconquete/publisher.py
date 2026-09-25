"""Publishing the game shako actually plays, so that it can be watched and recorded.

The bridge shows a game in two places — the live channel the game's page follows with
`?spectate=<port>`, and the file its "Replay a Game" opens — and both show only what a
client `publish`es. `shako-bridge.md` is explicit about why: a search applies thousands of
actions per move, and the bridge cannot tell which one was retained.

## Why the adapter's own states cannot be published

`AotReconqueteAdapter` is stateless: every call uploads its state with `from_save`, asks its
question, and releases the handle. A state born of `from_save` carries an **empty journal** —
`seed: null`, its own position as `initialState`, no action. Publishing one per move would
make the page reload the whole board at every move instead of animating it, and would leave
a recording holding the last position and not one move: nothing to replay.

## A mirror line

So this module keeps **one** line alive in the bridge, beside the adapter's throwaway ones.
The game's initial state is uploaded once; then, for every move the engine really played —
never a search's simulations — that move is applied to the line, the new state is published,
and the previous one is released. The journal of that line *is* the game: its initial state,
and every move in order. The generator's state rides in the state, so the mirror lands
exactly on the positions shako reached; this is checked at every move rather than assumed.

## Where it plugs in

`MirrorPublisher` is `SimulationEngine`'s `on_action_applied` hook. The engine is the only
place that knows which action was applied — after an illegal or timed-out choice was
replaced, and only once `apply_action` succeeded — so a move the game refused is never
published: the adapter raises `IllegalActionError` before the hook is called.
"""

from __future__ import annotations

from typing import Any, Self

from core.types import Action, State
from games.aot_reconquete.adapter import (
    AotReconqueteAdapter,
    IllegalActionError,
    _changed,
)
from games.aot_reconquete.bridge import AotReconqueteError, ShakoBridge

# Stamped afresh at every serialisation — the one field two states of one position differ by.
_VOLATILE_SAVE_FIELDS = frozenset({"savedAt"})


class MirrorDivergedError(AotReconqueteError):
    """The mirror line reached another position than the game shako played.

    It should not happen — the mirror applies the same action to the same position with the
    same generator state — so it is raised rather than papered over by republishing: a page
    and a recording showing another game than the one played would be worse than no page.
    """


class MirrorPublisher:
    """Keeps one line of play in the bridge that follows the engine's, and publishes it.

    Use it as the engine's hook, with the adapter the engine plays — its bridge is the one
    serving the channel and writing the file::

        adapter = AotReconqueteAdapter(live_port=8090, record_file="game.json")
        with MirrorPublisher(adapter) as publisher:
            SimulationEngine(adapter, [agent], on_action_applied=publisher).run_game()

    A game starts a new line by itself: when the position a move is played from is not the
    one the mirror stands on — the first move of a game, the first move of the next game of a
    batch — the line restarts from it, and the page reloads. Across a whole batch the
    recording is therefore the batch's last game.
    """

    def __init__(self, adapter: AotReconqueteAdapter) -> None:
        self.adapter = adapter
        self._bridge: ShakoBridge | None = None
        self._handle: str | None = None
        # The mirror's position as last read from the bridge, so that telling a move that
        # continues the line from one that starts a new game costs no round trip.
        self._position: dict[str, Any] | None = None
        # What the last `publish` answered: how many moves the published line holds.
        self.published_actions = 0

    # ---------------------------------------------------------- the engine hook

    def __call__(self, state: State, action: Action, player_id: int, next_state: State) -> None:
        """Play `action` on the mirror, check it reached `next_state`, and publish it."""
        bridge = self.adapter.bridge()
        if bridge is not self._bridge or self._position is None or not _same_position(
            self._position, state.data
        ):
            self._start(bridge, state)
        assert self._handle is not None and self._position is not None

        applied = bridge.call("apply_action", state=self._handle, action=action.data)["state"]
        try:
            reached = bridge.call("get_observable_state", state=applied)["observed"]
            # Checked here as well as in the adapter: a refused action comes back as the same
            # position (AOT-175), and publishing that would put a move in the journal that the
            # game never played.
            if not _changed(self._position, reached):
                raise IllegalActionError(
                    f"the game refused {self.adapter.get_action_display(action)} on the "
                    f"mirror line; it was not published"
                )
            if not _same_position(reached, next_state.data):
                raise MirrorDivergedError(
                    f"after {self.adapter.get_action_display(action)} the mirror line is not "
                    f"at the position shako played; nothing was published"
                )
            self.published_actions = int(bridge.call("publish", state=applied)["actions"])
        except BaseException:
            bridge.call("release_state", state=applied)
            raise
        bridge.call("release_state", state=self._handle)
        self._handle = applied
        self._position = reached

    # ------------------------------------------------------------ inspection

    def journal(self) -> dict[str, Any]:
        """The mirror line's journal — `{"seed", "initialState", "actions"}` — as the bridge
        answers it, and as the recording file and the channel's `game` event carry it."""
        return self._require_line().call("journal", state=self._handle)

    def mirror_state(self) -> State:
        """The mirror's position, read afresh from the bridge."""
        bridge = self._require_line()
        return State(data=bridge.call("get_observable_state", state=self._handle)["observed"])

    # ------------------------------------------------------------- life cycle

    def close(self) -> None:
        """Release the mirror line. What was published stays published, and recorded."""
        if self._handle is not None and self._bridge is not None and self._bridge.is_running():
            self._bridge.call("release_state", state=self._handle)
        self._bridge = None
        self._handle = None
        self._position = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        """Refused: the line is a handle in one bridge process, and means nothing in another.

        This is what `run_batch(n_workers > 1)` meets when it pickles an engine carrying this
        hook — a batch across processes cannot be watched; see `adapter.py`.
        """
        raise AotReconqueteError(
            "a MirrorPublisher follows one line in one bridge process and cannot be copied into "
            "another: play the published game in this process — run_game(), or "
            "run_batch(n_workers=1)."
        )

    # ------------------------------------------------------------- internals

    def _start(self, bridge: ShakoBridge, state: State) -> None:
        """Begin a new line at `state` — the only `from_save` the line ever makes."""
        self.close()
        self._handle = bridge.call("from_save", game=state.data["game"], rng=state.data["rng"])[
            "state"
        ]
        self._bridge = bridge
        self._position = state.data

    def _require_line(self) -> ShakoBridge:
        if self._handle is None or self._bridge is None:
            raise AotReconqueteError("no line yet: the engine has not played a move")
        return self._bridge


def _same_position(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Equal states, but for the save's timestamp. Arrays keep their order: the mirror must
    be the very same state, not one that merely describes the same board."""
    return _without_volatile(a) == _without_volatile(b)


def _without_volatile(state: dict[str, Any]) -> dict[str, Any]:
    game = state.get("game")
    if not isinstance(game, dict):
        return state
    return {
        **state,
        "game": {k: v for k, v in game.items() if k not in _VOLATILE_SAVE_FIELDS},
    }
