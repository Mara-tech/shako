"""Tests for the AOT Reconquête adapter, against the real TypeScript game.

They launch the game's Node bridge for real. There is no mock of it on purpose: what this
adapter is, is the mapping onto a protocol, and a fake that answered the way this file
expects would prove the fake right, not the mapping. The whole file skips when the game
repository is not installed next to shako (see `bridge.resolve_game_dir`).
"""

from __future__ import annotations

import pickle
import shutil
import subprocess
import time
from collections.abc import Iterator

import pytest

from core.engine import SimulationEngine
from core.types import Action, State
from games.aot_reconquete.adapter import AotReconqueteAdapter, IllegalActionError
from games.aot_reconquete.bridge import (
    AotReconqueteError,
    BridgeCallError,
    is_game_available,
)
from rl.random_agent import RandomAgent

pytestmark = pytest.mark.skipif(
    not is_game_available(),
    reason="no aot-reconquete.js checkout with its dependencies installed (see rules.md)",
)

# Pins the game every test plays, so a failure is replayable. The adapter passes it to the
# bridge, which builds the same board and rolls the same scouts from it.
SEED = 7

# A well-formed move nobody can play from the opening position: no force stands on (0, 0).
ILLEGAL_MOVE = Action(
    data={"kind": "move", "from": {"col": 0, "row": 0}, "to": {"col": 0, "row": 1}}
)


@pytest.fixture(scope="module")
def adapter() -> Iterator[AotReconqueteAdapter]:
    """One Node process for the whole module — the adapter is stateless, so it can be shared."""
    with AotReconqueteAdapter(seed=SEED) as shared:
        yield shared


@pytest.fixture(scope="module")
def initial(adapter: AotReconqueteAdapter) -> State:
    return adapter.get_initial_state()


def _bridge_process_count() -> int:
    """How many game bridges are running right now, as the OS sees them.

    0 where there is no `pgrep` to ask; the one assertion that reads this is guarded on it.
    """
    if not shutil.which("pgrep"):
        return 0
    found = subprocess.run(
        ["pgrep", "-f", "shakoBridgeMain"], capture_output=True, text=True, check=False
    )
    return len([line for line in found.stdout.split() if line.strip()])


def _wait_for_bridge_count(at_most: int, timeout_s: float = 10.0) -> int:
    """Poll until no more than `at_most` bridges are left, or give up and report what is.

    A bridge orphaned by a killed worker ends on the EOF of its stdin, which is the protocol's
    own shutdown — but it ends a moment *after* the process that owned it, not with it.
    Measured at ~60 ms on 2026-09-13, so this waits rather than asserting into the race.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        running = _bridge_process_count()
        if running <= at_most or time.monotonic() > deadline:
            return running
        time.sleep(0.05)


# -------- the shape of a one-player, perfect-information game ------------------


def test_one_player_who_never_changes(adapter: AotReconqueteAdapter, initial: State) -> None:
    assert adapter.get_n_players() == 1
    assert adapter.get_current_player(initial) == 0


def test_observable_state_is_the_whole_state(
    adapter: AotReconqueteAdapter, initial: State
) -> None:
    observed = adapter.get_observable_state(initial, 0)
    assert observed.player_id == 0
    assert observed.data == initial.data


def test_initial_state_carries_the_save_and_the_dice(initial: State) -> None:
    assert sorted(initial.data) == ["game", "rng"]
    assert isinstance(initial.data["rng"], int)
    # The save, as the game's own "Save game" writes it — carried through unchanged.
    assert initial.data["game"]["turnCounter"] == 1
    assert initial.data["game"]["dimensions"] == {"columns": 9, "rows": 12}


def test_a_fresh_game_is_not_over(adapter: AotReconqueteAdapter, initial: State) -> None:
    assert adapter.is_terminal(initial) is False
    assert adapter.get_scores(initial) == {0: 0.0}


# -------- legal actions -------------------------------------------------------


def test_legal_actions_are_not_empty_on_a_live_position(
    adapter: AotReconqueteAdapter, initial: State
) -> None:
    legal = adapter.get_legal_actions(initial, 0)
    assert legal
    assert all(isinstance(action, Action) and "kind" in action.data for action in legal)
    assert {"kind": "end_of_turn"} in [action.data for action in legal]


def test_only_the_end_of_turn_remains_when_no_force_can_act(
    adapter: AotReconqueteAdapter, initial: State
) -> None:
    """The case `BaseAdapter`'s invariant is really about: nobody can act, the game goes on.

    Every force spends its action points on side acts and moves — attacks excluded, so the
    game stays out of combat — until nothing but the end of turn is left. The list must
    shrink to exactly one action and never to none.
    """
    state = initial
    for _ in range(60):
        legal = adapter.get_legal_actions(state, 0)
        assert legal, "no legal action on a non-terminal state"
        playable = [a for a in legal if a.data["kind"] not in ("end_of_turn", "attack")]
        if not playable:
            assert [a.data for a in legal] == [{"kind": "end_of_turn"}]
            return
        state = adapter.apply_action(state, playable[0], 0)
        assert not adapter.is_terminal(state), "the game ended before the forces were spent"
    pytest.fail("the forces were still acting after 60 actions")


# -------- statelessness -------------------------------------------------------


def test_apply_action_does_not_touch_the_state_it_is_given(
    adapter: AotReconqueteAdapter, initial: State
) -> None:
    before = pickle.dumps(initial.data)
    successor = adapter.apply_action(initial, adapter.get_legal_actions(initial, 0)[0], 0)
    assert pickle.dumps(initial.data) == before, "apply_action mutated the state it was given"
    assert successor.data != initial.data
    assert successor is not initial


def test_the_same_state_can_be_played_from_twice(
    adapter: AotReconqueteAdapter, initial: State
) -> None:
    """What a search does all day: hold a node and explore several moves out of it."""
    legal = adapter.get_legal_actions(initial, 0)
    first = adapter.apply_action(initial, legal[0], 0)
    second = adapter.apply_action(initial, legal[0], 0)
    assert first.data["game"]["scoutForces"] == second.data["game"]["scoutForces"]
    assert adapter.get_legal_actions(initial, 0) == legal, "the source position moved"


def test_clone_state_is_independent(adapter: AotReconqueteAdapter, initial: State) -> None:
    clone = adapter.clone_state(initial)
    assert clone.data == initial.data
    clone.data["game"]["turnCounter"] = 999
    clone.data["game"]["terrain"][0]["kind"] = "nonsense"
    assert initial.data["game"]["turnCounter"] == 1
    assert initial.data["game"]["terrain"][0]["kind"] != "nonsense"


def test_clone_state_never_reaches_the_bridge(initial: State) -> None:
    """It is called in a loop by MCTS, self-play and greedy, so it must cost no round trip.

    An adapter that has never launched a process clones all the same, and still has none
    afterwards.
    """
    never_started = AotReconqueteAdapter()
    assert never_started._bridge is None
    assert never_started.clone_state(initial).data == initial.data
    assert never_started._bridge is None, "clone_state launched the game"


# -------- a game, end to end --------------------------------------------------


def test_a_scripted_game_reaches_a_terminal_state(adapter: AotReconqueteAdapter) -> None:
    """Ending every turn at once lets the titans come, and the game ends — in a defeat here.

    A short game on purpose: it is the turn loop, the end of the game and the scores that are
    under test, not the quality of the play.
    """
    state = adapter.get_initial_state()
    played = 0
    while not adapter.is_terminal(state):
        legal = adapter.get_legal_actions(state, 0)
        assert legal
        end_of_turn = [a for a in legal if a.data["kind"] == "end_of_turn"]
        state = adapter.apply_action(state, end_of_turn[0] if end_of_turn else legal[0], 0)
        played += 1
        assert played < 300, "the game went on for ever"

    assert played > 0
    assert adapter.get_legal_actions(state, 0) == [], "a terminal state offers no action"
    scores = adapter.get_scores(state)
    assert set(scores) == {0}
    assert 0.0 <= scores[0] <= 1.0


def test_the_engine_plays_a_game_with_a_random_agent(adapter: AotReconqueteAdapter) -> None:
    """The whole stack: engine turn loop, agent, adapter, pipe. Capped to stay a test."""
    result = SimulationEngine(adapter, [RandomAgent(seed=1)], max_turns=12).run_game()
    assert result.n_turns == 12
    assert set(result.scores) == {0}
    # Every action the agent played came from get_legal_actions, so none was replaced.
    assert result.illegal_action_counts == {0: 0}


# -------- what a failure looks like -------------------------------------------


def test_an_illegal_action_raises_rather_than_doing_nothing(
    adapter: AotReconqueteAdapter, initial: State
) -> None:
    """The bridge answers an unchanged position; unsaid, that is a search looping in place."""
    with pytest.raises(IllegalActionError, match="refused"):
        adapter.apply_action(initial, ILLEGAL_MOVE, 0)


def test_an_action_the_game_does_not_name_raises(
    adapter: AotReconqueteAdapter, initial: State
) -> None:
    with pytest.raises(BridgeCallError) as raised:
        adapter.apply_action(initial, Action(data={"kind": "fly"}), 0)
    assert raised.value.code == "invalid_params"
    assert "kind" in str(raised.value)


def test_a_state_this_game_did_not_produce_raises(adapter: AotReconqueteAdapter) -> None:
    with pytest.raises(AotReconqueteError, match="carries"):
        adapter.is_terminal(State(data={"sticks": 21}))


def test_the_bridge_survives_what_it_refused(
    adapter: AotReconqueteAdapter, initial: State
) -> None:
    """A refusal is an answer, not a crash — the next question must still be served."""
    with pytest.raises(AotReconqueteError):
        adapter.apply_action(initial, Action(data={"kind": "fly"}), 0)
    assert adapter.bridge().is_running()
    assert adapter.get_legal_actions(initial, 0), "the bridge stopped answering after a refusal"


# -------- pickling, batches, and the process ----------------------------------


def test_the_adapter_survives_a_pickle_round_trip(initial: State) -> None:
    """`core/engine.py` pickles the adapter into spawned workers; a Popen cannot be pickled."""
    original = AotReconqueteAdapter(seed=SEED)
    original.get_initial_state()  # the process is running, so it is in the way of the pickle
    try:
        restored = pickle.loads(pickle.dumps(original))
    finally:
        original.close()

    assert restored._bridge is None, "the unpickled adapter carried a process handle"
    assert restored.seed == SEED
    try:
        rebuilt = restored.get_initial_state()
    finally:
        restored.close()

    # The same seed builds the same game — compared on the position rather than on the whole
    # dict, `game.savedAt` being stamped afresh at every serialisation.
    assert rebuilt.data["rng"] == initial.data["rng"]
    assert rebuilt.data["game"]["scoutForces"] == initial.data["game"]["scoutForces"]
    assert rebuilt.data["game"]["titanSwarms"] == initial.data["game"]["titanSwarms"]


def test_run_batch_across_several_processes(adapter: AotReconqueteAdapter) -> None:
    """The case that breaks without `__getstate__`: spawn, one bridge per worker."""
    bridges_before = _bridge_process_count()
    batched = AotReconqueteAdapter(seed=SEED)
    engine = SimulationEngine(batched, [RandomAgent(seed=3)], max_turns=3)
    results = engine.run_batch(n_games=4, n_workers=2)

    assert len(results) == 4
    assert all(result.n_turns == 3 for result in results)
    batched.close()
    if shutil.which("pgrep"):
        assert _wait_for_bridge_count(bridges_before) <= bridges_before, (
            "a bridge outlived the batch: a worker killed by the pool left a Node process "
            "behind instead of closing its pipe"
        )


def test_closing_the_adapter_stops_the_process() -> None:
    standalone = AotReconqueteAdapter(seed=SEED)
    standalone.get_initial_state()
    bridge = standalone.bridge()
    assert bridge.is_running()

    standalone.close()
    assert not bridge.is_running(), "the Node process outlived the adapter that owns it"
    # And it comes back: closing is not a one-way door.
    assert standalone.get_initial_state().data["game"]["turnCounter"] == 1
    standalone.close()


# -------- the analyzer's view -------------------------------------------------


def test_action_labels_are_the_coarse_kind(adapter: AotReconqueteAdapter, initial: State) -> None:
    """Every move names two tiles, so the full action would make each one its own category."""
    labels = {adapter.get_action_label(a) for a in adapter.get_legal_actions(initial, 0)}
    assert labels <= {
        "move", "join_force", "attack", "end_of_turn", "heal_rest", "change_gas",
        "change_blades", "repair_door", "build_outpost", "find_horse", "leave_horse",
        "combat_attack", "leave_combat", "capture_titan",
    }
    assert "end_of_turn" in labels
