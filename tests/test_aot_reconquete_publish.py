"""Tests for publishing the game shako plays — the live channel and the recording.

Against the real bridge, as `test_aot_reconquete.py` is and for the same reason: what is under
test is what the game's own process receives, holds and writes, and a fake of it would prove
the fake. The file skips without the game; the recording tests also skip on a checkout that
predates the recorder (AOT-170).
"""

from __future__ import annotations

import http.client
import json
import pickle
import queue
import socket
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from core.base_agent import BaseAgent
from core.engine import SimulationEngine
from core.types import Action, ObservableState, State
from games.aot_reconquete.adapter import AotReconqueteAdapter, IllegalActionError
from games.aot_reconquete.bridge import (
    LIVE_PORT_ENV_VAR,
    RECORD_FILE_ENV_VAR,
    AotReconqueteError,
    BridgeProcessError,
    ShakoBridge,
    is_game_available,
    resolve_game_dir,
)
from games.aot_reconquete.publisher import MirrorPublisher
from rl.mcts_agent import MCTSAgent
from rl.random_agent import RandomAgent

pytestmark = pytest.mark.skipif(
    not is_game_available(),
    reason="no aot-reconquete.js checkout with its dependencies installed (see rules.md)",
)

needs_recorder = pytest.mark.skipif(
    not is_game_available()
    or not (resolve_game_dir() / "src" / "game" / "headless" / "GameRecorder.ts").is_file(),
    reason="the game checkout predates SHAKO_RECORD_FILE (AOT-170)",
)

SEED = 7
MOVES = 15

# A well-formed move nobody can play from the opening position: no force stands on (0, 0).
ILLEGAL_MOVE = Action(
    data={"kind": "move", "from": {"col": 0, "row": 0}, "to": {"col": 0, "row": 1}}
)


class _Spy:
    """An `on_action_applied` hook that notes every call, then hands it to the publisher."""

    def __init__(self, publisher: MirrorPublisher) -> None:
        self.publisher = publisher
        self.calls: list[tuple[State, Action, int, State]] = []

    def __call__(self, state: State, action: Action, player_id: int, next_state: State) -> None:
        self.calls.append((state, action, player_id, next_state))
        self.publisher(state, action, player_id, next_state)


class _AlwaysIllegalAgent(BaseAgent):
    """Returns an action outside the legal list, so the engine replaces every one of them."""

    def on_game_start(self, player_id: int, n_players: int) -> None:
        pass

    def on_game_end(self, scores: dict[int, float]) -> None:
        pass

    def choose_action(self, observable_state: ObservableState, legal: list[Action]) -> Action:
        return Action(data={"kind": "never_legal"})


@pytest.fixture
def adapter() -> Iterator[AotReconqueteAdapter]:
    with AotReconqueteAdapter(seed=SEED) as fresh:
        yield fresh


def _played(result: Any) -> list[dict[str, Any]]:
    return [action.data for _, _, action in result.actions]


def _without_saved_at(state: dict[str, Any]) -> dict[str, Any]:
    return {**state, "game": {k: v for k, v in state["game"].items() if k != "savedAt"}}


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


# -------- the published line is the game played --------------------------------


def test_the_published_journal_is_exactly_the_game_the_engine_played(
    adapter: AotReconqueteAdapter,
) -> None:
    """Every action, in order, from the game's initial state — and nothing else."""
    with MirrorPublisher(adapter) as publisher:
        spy = _Spy(publisher)
        result = SimulationEngine(
            adapter, [RandomAgent(seed=1)], max_turns=MOVES, record=True, on_action_applied=spy
        ).run_game()
        journal = publisher.journal()

    assert result.n_turns == MOVES
    assert journal["actions"] == _played(result)
    assert _without_saved_at(journal["initialState"]) == _without_saved_at(spy.calls[0][0].data)
    assert journal["seed"] is None, "the line starts at a from_save, as the protocol says"
    assert publisher.published_actions == MOVES


def test_the_mirror_stands_on_shakos_position_after_every_move(
    adapter: AotReconqueteAdapter,
) -> None:
    """The generator's state travels in the state, so the mirror is the very same game —
    compared whole, arrays in their order, but for the save's timestamp."""
    with MirrorPublisher(adapter) as publisher:
        mirrored: list[tuple[State, State]] = []

        def check(state: State, action: Action, player_id: int, next_state: State) -> None:
            publisher(state, action, player_id, next_state)
            mirrored.append((publisher.mirror_state(), next_state))

        SimulationEngine(
            adapter, [RandomAgent(seed=2)], max_turns=MOVES, on_action_applied=check
        ).run_game()

    assert len(mirrored) == MOVES
    for index, (mirror, played) in enumerate(mirrored):
        assert _without_saved_at(mirror.data) == _without_saved_at(played.data), (
            f"the mirror left the game at move {index}"
        )


def test_a_replaced_choice_is_published_as_the_move_played(
    adapter: AotReconqueteAdapter,
) -> None:
    """The engine replaces an illegal choice with a random legal action; the replacement is
    what was played, so it is what the page and the recording must show."""
    with MirrorPublisher(adapter) as publisher:
        result = SimulationEngine(
            adapter, [_AlwaysIllegalAgent()], max_turns=5, record=True, on_action_applied=publisher
        ).run_game()
        journal = publisher.journal()

    assert result.illegal_action_counts == {0: 5}
    assert journal["actions"] == _played(result)
    assert all(action["kind"] != "never_legal" for action in journal["actions"])


def test_a_refused_move_is_not_published(adapter: AotReconqueteAdapter) -> None:
    """The engine never reaches the hook with a refused move — the adapter raises first — and
    the publisher refuses one handed to it directly all the same.

    Not a formality: the bridge *does* journal a move its game refused (the position is
    unchanged, the action is appended), and publishes it if asked. Only this check keeps it
    out of the page and the recording.
    """
    start = adapter.get_initial_state()
    first = adapter.get_legal_actions(start, 0)[0]
    after = adapter.apply_action(start, first, 0)
    with MirrorPublisher(adapter) as publisher:
        publisher(start, first, 0, after)
        before = publisher.journal()

        refused: IllegalActionError | None = None
        try:
            publisher(after, ILLEGAL_MOVE, 0, after)
        except IllegalActionError as raised:
            refused = raised

        assert publisher.journal() == before, "a refused move reached the published line"
        assert publisher.published_actions == 1
        assert refused is not None and "not published" in str(refused)
        # The line is intact: the next real move still continues it.
        second = adapter.get_legal_actions(after, 0)[0]
        publisher(after, second, 0, adapter.apply_action(after, second, 0))
        assert publisher.journal()["actions"] == [first.data, second.data]


def test_a_new_game_starts_a_new_line() -> None:
    """Two games in one process: the second restarts the mirror rather than diverging."""
    unseeded = AotReconqueteAdapter()
    try:
        with MirrorPublisher(unseeded) as publisher:
            spy = _Spy(publisher)
            engine = SimulationEngine(
                unseeded, [RandomAgent(seed=3)], max_turns=4, record=True, on_action_applied=spy
            )
            results = engine.run_batch(n_games=2, n_workers=1)
            journal = publisher.journal()
    finally:
        unseeded.close()

    assert journal["actions"] == _played(results[1])
    second_start = spy.calls[4][0].data
    assert _without_saved_at(journal["initialState"]) == _without_saved_at(second_start)
    assert _without_saved_at(spy.calls[0][0].data) != _without_saved_at(second_start)


# -------- the recording --------------------------------------------------------


@needs_recorder
def test_the_recording_is_the_published_journal(tmp_path: Path) -> None:
    record = tmp_path / "game.json"
    with AotReconqueteAdapter(seed=SEED, record_file=str(record)) as adapter:
        with MirrorPublisher(adapter) as publisher:
            spy = _Spy(publisher)
            result = SimulationEngine(
                adapter, [RandomAgent(seed=4)], max_turns=MOVES, record=True, on_action_applied=spy
            ).run_game()
            journal = publisher.journal()

    recorded = json.loads(record.read_text(encoding="utf-8"))
    assert recorded == journal
    assert recorded["actions"] == _played(result)
    assert _without_saved_at(recorded["initialState"]) == _without_saved_at(spy.calls[0][0].data)


@needs_recorder
def test_nothing_is_recorded_until_a_move_is_published(tmp_path: Path) -> None:
    """The adapter's own states are never published — the file waits for the mirror."""
    record = tmp_path / "game.json"
    with AotReconqueteAdapter(seed=SEED, record_file=str(record)) as adapter:
        state = adapter.get_initial_state()
        action = adapter.get_legal_actions(state, 0)[0]
        after = adapter.apply_action(state, action, 0)
        assert not record.exists()
        with MirrorPublisher(adapter) as publisher:
            publisher(state, action, 0, after)
    assert json.loads(record.read_text(encoding="utf-8"))["actions"] == [action.data]


@needs_recorder
def test_a_relative_record_file_is_taken_from_the_python_side(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bridge runs in the game repository; a relative path must not land there.

    Twice: the adapter resolves the path when it is built — a `chdir` before its bridge
    starts must not move the file — and `ShakoBridge` resolves one it is handed directly.
    """
    name = f"relative-{tmp_path.name}.json"
    built_in, moved_to, direct = tmp_path / "built", tmp_path / "moved", tmp_path / "direct"
    for folder in (built_in, moved_to, direct):
        folder.mkdir()

    monkeypatch.chdir(built_in)
    adapter = AotReconqueteAdapter(seed=SEED, record_file=name)
    monkeypatch.chdir(moved_to)
    with adapter, MirrorPublisher(adapter) as publisher:
        SimulationEngine(
            adapter, [RandomAgent(seed=5)], max_turns=2, on_action_applied=publisher
        ).run_game()

    monkeypatch.chdir(direct)
    with ShakoBridge(record_file=name) as bridge:
        start = bridge.call("get_initial_state", seed=SEED)["state"]
        action = bridge.call("get_legal_actions", state=start)["actions"][0]
        bridge.call("publish", state=bridge.call("apply_action", state=start, action=action)["state"])

    assert (built_in / name).is_file()
    assert not (moved_to / name).exists()
    assert (direct / name).is_file()
    assert not (resolve_game_dir() / name).exists()


# -------- the live channel -----------------------------------------------------


def _read_frames(port: int, frames: queue.Queue[dict[str, Any]], connected: threading.Event) -> None:
    """Follow `GET /live` as a page would, and queue every event's payload."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        connection.request("GET", "/live")
        response = connection.getresponse()
        assert response.status == 200
        connected.set()
        event = None
        while True:
            line = response.readline().decode("utf-8")
            if line == "":
                return
            line = line.rstrip("\r\n")
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                payload = json.loads(line[len("data: "):])
                assert payload["kind"] == event, "the frame's name and its payload disagree"
                frames.put(payload)
    except (OSError, http.client.HTTPException):
        return
    finally:
        connection.close()


def test_the_live_channel_streams_the_moves_played() -> None:
    """A watcher connected before the game gets the game, then one `action` per move — the
    same moves, in order.

    Played by MCTS on the very bridge that serves the channel, so it is also the test that the
    search's own `apply_action`s never reach a watcher: one of them published would arrive as a
    `game` event mid-game, the page reloading on a line nobody played.
    """
    with AotReconqueteAdapter(seed=SEED, live_port=0) as adapter:
        port = adapter.bridge().bound_live_port
        assert port
        frames: queue.Queue[dict[str, Any]] = queue.Queue()
        connected = threading.Event()
        threading.Thread(target=_read_frames, args=(port, frames, connected), daemon=True).start()
        assert connected.wait(10), "the channel did not answer"

        agent = MCTSAgent(adapter, n_simulations=4, max_rollout_depth=2, seed=1)
        with MirrorPublisher(adapter) as publisher:
            result = SimulationEngine(
                adapter, [agent], max_turns=6, record=True, on_action_applied=publisher
            ).run_game()

        received = [frames.get(timeout=10)]
        assert received[0]["kind"] == "game"
        watched = list(received[0]["journal"]["actions"])
        while len(watched) < result.n_turns:
            frame = frames.get(timeout=10)
            assert frame["kind"] == "action", "the page was told to reload mid-game"
            assert frame["index"] == len(watched)
            watched.append(frame["action"])

    assert watched == _played(result)


def test_a_port_already_taken_fails_at_start_with_the_bridges_reason() -> None:
    with AotReconqueteAdapter(seed=SEED, live_port=0) as first:
        taken = first.bridge().bound_live_port
        second = ShakoBridge(live_port=taken)
        with pytest.raises(BridgeProcessError, match="EADDRINUSE"):
            second.start()
        second.close()


def test_the_inherited_environment_is_not_the_bridges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With neither parameter, the bridge serves nothing — whatever the shell exported.

    The unreadable port is the sharp end: a bridge that inherited it would refuse to start.
    """
    stray = tmp_path / "stray.json"
    monkeypatch.setenv(LIVE_PORT_ENV_VAR, "not-a-port")
    monkeypatch.setenv(RECORD_FILE_ENV_VAR, str(stray))
    with AotReconqueteAdapter(seed=SEED) as adapter:
        state = adapter.get_initial_state()
        action = adapter.get_legal_actions(state, 0)[0]
        with MirrorPublisher(adapter) as publisher:
            publisher(state, action, 0, adapter.apply_action(state, action, 0))
        assert adapter.bridge().bound_live_port is None
    assert not stray.exists()


def test_an_invalid_live_port_is_refused_before_launching() -> None:
    for wrong in (-1, 65536, True, "8090"):
        with pytest.raises(ValueError, match="live_port"):
            ShakoBridge(live_port=wrong)  # type: ignore[arg-type]


# -------- several processes: refused ------------------------------------------


def test_a_publishing_adapter_refuses_to_be_copied_into_a_worker(tmp_path: Path) -> None:
    for adapter in (
        AotReconqueteAdapter(live_port=_free_port()),
        AotReconqueteAdapter(record_file=str(tmp_path / "game.json")),
    ):
        with pytest.raises(AotReconqueteError, match="n_workers=1"):
            pickle.dumps(adapter)
    assert pickle.loads(pickle.dumps(AotReconqueteAdapter())).live_port is None


def test_run_batch_across_processes_is_refused_when_publishing(tmp_path: Path) -> None:
    """Refused before a game is played, with the reason — not a port conflict in a worker."""
    recording = AotReconqueteAdapter(seed=SEED, record_file=str(tmp_path / "game.json"))
    with pytest.raises(AotReconqueteError, match="cannot be copied"):
        SimulationEngine(recording, [RandomAgent(seed=6)], max_turns=2).run_batch(
            n_games=2, n_workers=2
        )
    assert not (tmp_path / "game.json").exists()

    plain = AotReconqueteAdapter(seed=SEED)
    try:
        with MirrorPublisher(plain) as publisher:
            engine = SimulationEngine(
                plain, [RandomAgent(seed=6)], max_turns=2, on_action_applied=publisher
            )
            with pytest.raises(AotReconqueteError, match="MirrorPublisher"):
                engine.run_batch(n_games=2, n_workers=2)
    finally:
        plain.close()
