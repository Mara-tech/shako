from __future__ import annotations

import pytest

from core.engine import SimulationEngine
from core.types import Action, State
from games.seven_families.adapter import SevenFamiliesAdapter
from rl.random_agent import RandomAgent


# ------------------------------------------------------------------ helpers

def _adapter(n_players: int = 3, n_families: int = 3, n_components: int = 3, seed: int = 0) -> SevenFamiliesAdapter:
    return SevenFamiliesAdapter(n_players=n_players, n_families=n_families, n_components=n_components, seed=seed)


def _card(family: int, component: int, n_components: int = 3) -> int:
    return family * n_components + component


def _state(
    hands: dict[int, list[int]],
    deck: list[int] | None = None,
    books: dict[int, list[int]] | None = None,
    current: int = 0,
    game_over: bool = False,
    n_players: int | None = None,
) -> State:
    n = n_players if n_players is not None else len(hands)
    return State(
        data={
            "hands": {pid: list(h) for pid, h in hands.items()},
            "deck": list(deck) if deck is not None else [],
            "books": books if books is not None else {pid: [] for pid in range(n)},
            "current": current,
            "game_over": game_over,
        }
    )


def _ask(target: int, family: int, component: int) -> Action:
    return Action(data={"target": target, "family": family, "component": component})


def _pass() -> Action:
    return Action(data={"pass": True})


# ------------------------------------------------------------------ initial state

def test_initial_state_all_cards_accounted_for() -> None:
    a = _adapter(n_players=4, n_families=7, n_components=6)
    state = a.get_initial_state()
    d = state.data
    total = 7 * 6
    in_hands = sum(len(h) for h in d["hands"].values())
    in_deck = len(d["deck"])
    booked_cards = sum(len(b) * 6 for b in d["books"].values())
    assert in_hands + in_deck + booked_cards == total


def test_initial_state_hand_sizes_even() -> None:
    a = _adapter(n_players=4, n_families=7, n_components=6)
    state = a.get_initial_state()
    d = state.data
    expected_hand = (7 * 6) // 4  # 10
    for pid in range(4):
        assert len(d["hands"][pid]) == expected_hand


def test_initial_state_no_duplicate_cards() -> None:
    a = _adapter(n_players=3, n_families=4, n_components=4, seed=1)
    state = a.get_initial_state()
    d = state.data
    all_cards = [c for h in d["hands"].values() for c in h] + d["deck"]
    assert len(all_cards) == len(set(all_cards))


def test_initial_state_game_not_over() -> None:
    a = _adapter()
    state = a.get_initial_state()
    assert not state.data["game_over"]


def test_initial_state_current_player_is_0() -> None:
    a = _adapter()
    state = a.get_initial_state()
    assert state.data["current"] == 0


# ------------------------------------------------------------------ legal actions

def test_legal_actions_empty_hand_returns_pass() -> None:
    a = _adapter()
    state = _state(hands={0: [], 1: [_card(0, 0)], 2: [_card(1, 0)]})
    legal = a.get_legal_actions(state, 0)
    assert len(legal) == 1
    assert legal[0].data.get("pass")


def test_legal_actions_only_families_in_hand() -> None:
    a = _adapter(n_players=2)
    # Player 0 has only family-0 cards; cannot ask for family 1 or 2
    state = _state(
        hands={0: [_card(0, 0)], 1: [_card(1, 0), _card(2, 0)]},
        n_players=2,
    )
    legal = a.get_legal_actions(state, 0)
    families_asked = {act.data["family"] for act in legal}
    assert families_asked == {0}


def test_legal_actions_excludes_owned_components() -> None:
    a = _adapter(n_players=2)
    # Player 0 already has component 0 of family 0; should only ask for 1 and 2
    state = _state(
        hands={0: [_card(0, 0)], 1: [_card(0, 1), _card(0, 2)]},
        n_players=2,
    )
    legal = a.get_legal_actions(state, 0)
    components_asked = {act.data["component"] for act in legal}
    assert 0 not in components_asked
    assert components_asked == {1, 2}


def test_legal_actions_targets_all_opponents() -> None:
    a = _adapter(n_players=3)
    state = _state(
        hands={0: [_card(0, 0)], 1: [_card(0, 1)], 2: [_card(0, 2)]},
        n_players=3,
    )
    legal = a.get_legal_actions(state, 0)
    targets = {act.data["target"] for act in legal}
    assert targets == {1, 2}


def test_legal_actions_does_not_target_self() -> None:
    a = _adapter(n_players=3)
    state = _state(
        hands={0: [_card(0, 0)], 1: [_card(1, 0)], 2: [_card(2, 0)]},
        n_players=3,
    )
    for act in a.get_legal_actions(state, 0):
        if not act.data.get("pass"):
            assert act.data["target"] != 0


# ------------------------------------------------------------------ apply_action: hit

def test_hit_transfers_card_to_asker() -> None:
    a = _adapter(n_players=2)
    state = _state(
        hands={0: [_card(0, 0)], 1: [_card(0, 1), _card(0, 2)]},
        n_players=2,
    )
    new = a.apply_action(state, _ask(1, 0, 1), player_id=0)
    assert _card(0, 1) in new.data["hands"][0]
    assert _card(0, 1) not in new.data["hands"][1]


def test_hit_asker_goes_again() -> None:
    a = _adapter(n_players=2, n_components=4)
    # Player 0 has one card of family 0; gets another — not yet complete (4 needed)
    state = _state(
        hands={0: [_card(0, 0, 4)], 1: [_card(0, 1, 4), _card(0, 2, 4)]},
        n_players=2,
    )
    new = a.apply_action(state, _ask(1, 0, 1), player_id=0)
    assert new.data["current"] == 0


def test_hit_does_not_mutate_source_state() -> None:
    a = _adapter(n_players=2)
    state = _state(
        hands={0: [_card(0, 0)], 1: [_card(0, 1), _card(0, 2)]},
        n_players=2,
    )
    original_hand_0 = list(state.data["hands"][0])
    original_hand_1 = list(state.data["hands"][1])
    a.apply_action(state, _ask(1, 0, 1), player_id=0)
    assert state.data["hands"][0] == original_hand_0
    assert state.data["hands"][1] == original_hand_1


# ------------------------------------------------------------------ apply_action: miss

def test_miss_draws_from_deck() -> None:
    a = _adapter(n_players=2)
    extra_card = _card(2, 0)
    state = _state(
        hands={0: [_card(0, 0)], 1: [_card(1, 0)]},
        deck=[extra_card],
        n_players=2,
    )
    new = a.apply_action(state, _ask(1, 0, 1), player_id=0)
    assert extra_card in new.data["hands"][0]
    assert len(new.data["deck"]) == 0


def test_miss_empty_deck_no_draw() -> None:
    a = _adapter(n_players=2)
    state = _state(
        hands={0: [_card(0, 0)], 1: [_card(1, 0)]},
        deck=[],
        n_players=2,
    )
    hand_before = len(state.data["hands"][0])
    new = a.apply_action(state, _ask(1, 0, 1), player_id=0)
    assert len(new.data["hands"][0]) == hand_before


def test_miss_advances_turn() -> None:
    a = _adapter(n_players=2)
    state = _state(
        hands={0: [_card(0, 0)], 1: [_card(1, 0)]},
        n_players=2,
    )
    new = a.apply_action(state, _ask(1, 0, 1), player_id=0)
    assert new.data["current"] == 1


# ------------------------------------------------------------------ book check

def test_complete_family_is_booked() -> None:
    a = _adapter(n_players=2, n_components=2)
    # Player 0 has one card and asks for the second — should complete family 0
    state = _state(
        hands={0: [_card(0, 0, 2)], 1: [_card(0, 1, 2)]},
        n_players=2,
    )
    new = a.apply_action(state, _ask(1, 0, 1), player_id=0)
    assert 0 in new.data["books"][0]
    assert len(new.data["hands"][0]) == 0


def test_game_over_when_all_families_booked() -> None:
    a = _adapter(n_players=2, n_families=1, n_components=2)
    state = _state(
        hands={0: [_card(0, 0, 2)], 1: [_card(0, 1, 2)]},
        books={0: [], 1: []},
        n_players=2,
    )
    new = a.apply_action(state, _ask(1, 0, 1), player_id=0)
    assert new.data["game_over"]
    assert a.is_terminal(new)


def test_booking_empty_hand_advances_turn() -> None:
    # After booking its only family a player's hand is empty; turn should pass.
    a = _adapter(n_players=2, n_families=2, n_components=2)
    state = _state(
        hands={
            0: [_card(0, 0, 2)],                                # only family-0 card
            1: [_card(0, 1, 2), _card(1, 0, 2), _card(1, 1, 2)],
        },
        books={0: [], 1: []},
        n_players=2,
    )
    new = a.apply_action(state, _ask(1, 0, 1), player_id=0)
    # Player 0 booked family 0 but has no cards left → turn passes to player 1
    assert 0 in new.data["books"][0]
    assert new.data["hands"][0] == []
    assert new.data["current"] == 1


# ------------------------------------------------------------------ pass action

def test_pass_advances_turn() -> None:
    a = _adapter(n_players=3)
    state = _state(
        hands={0: [], 1: [_card(0, 0)], 2: [_card(1, 0)]},
        n_players=3,
    )
    new = a.apply_action(state, _pass(), player_id=0)
    assert new.data["current"] == 1


def test_pass_skips_empty_hands() -> None:
    a = _adapter(n_players=3)
    # Player 1 also has an empty hand; turn should skip to player 2
    state = _state(
        hands={0: [], 1: [], 2: [_card(0, 0)]},
        n_players=3,
    )
    new = a.apply_action(state, _pass(), player_id=0)
    assert new.data["current"] == 2


# ------------------------------------------------------------------ observable state

def test_observable_state_shows_own_hand() -> None:
    a = _adapter()
    state = a.get_initial_state()
    obs = a.get_observable_state(state, 0)
    assert sorted(obs.data["hands"][0]) == sorted(state.data["hands"][0])


def test_observable_state_hides_opponent_hands() -> None:
    a = _adapter()
    state = a.get_initial_state()
    obs = a.get_observable_state(state, 0)
    for pid in range(1, 3):
        assert obs.data["hands"][pid] == []


def test_observable_state_exposes_hand_sizes() -> None:
    a = _adapter()
    state = a.get_initial_state()
    obs = a.get_observable_state(state, 0)
    for pid in range(1, 3):
        assert obs.data["hand_sizes"][pid] == len(state.data["hands"][pid])


def test_observable_state_has_same_top_level_keys_as_full_state() -> None:
    """Observable state must share the keys that get_legal_actions / apply_action use."""
    a = _adapter()
    state = a.get_initial_state()
    obs = a.get_observable_state(state, 0)
    for key in ("hands", "deck", "books", "current", "game_over"):
        assert key in obs.data, f"missing key '{key}' in observable state"


def test_observable_state_hands_are_copy() -> None:
    a = _adapter()
    state = a.get_initial_state()
    obs = a.get_observable_state(state, 0)
    obs.data["hands"][0].clear()
    assert state.data["hands"][0]  # original unchanged


# ------------------------------------------------------------------ clone state

def test_clone_is_independent() -> None:
    a = _adapter()
    state = a.get_initial_state()
    clone = a.clone_state(state)
    clone.data["hands"][0].append(999)
    clone.data["books"][0].append(99)
    clone.data["deck"].append(998)
    assert 999 not in state.data["hands"][0]
    assert 99 not in state.data["books"][0]
    assert 998 not in state.data["deck"]


# ------------------------------------------------------------------ sample_state

def test_sample_state_all_cards_accounted_for() -> None:
    a = _adapter(n_players=3, n_families=3, n_components=3, seed=5)
    state = a.get_initial_state()
    obs = a.get_observable_state(state, 0)
    sampled = a.sample_state(obs)
    d = sampled.data
    total = 3 * 3
    in_hands = sum(len(h) for h in d["hands"].values())
    in_deck = len(d["deck"])
    booked_cards = sum(len(b) * 3 for b in d["books"].values())
    assert in_hands + in_deck + booked_cards == total


def test_sample_state_preserves_own_hand() -> None:
    a = _adapter(seed=7)
    state = a.get_initial_state()
    obs = a.get_observable_state(state, 0)
    sampled = a.sample_state(obs)
    assert sorted(sampled.data["hands"][0]) == sorted(obs.data["hands"][0])


def test_sample_state_respects_hand_sizes() -> None:
    a = _adapter(seed=3)
    state = a.get_initial_state()
    obs = a.get_observable_state(state, 0)
    sampled = a.sample_state(obs)
    for pid in range(1, 3):
        assert len(sampled.data["hands"][pid]) == obs.data["hand_sizes"][pid]


def test_sample_state_no_duplicates() -> None:
    a = _adapter(seed=2)
    state = a.get_initial_state()
    obs = a.get_observable_state(state, 0)
    sampled = a.sample_state(obs)
    all_cards = [c for h in sampled.data["hands"].values() for c in h] + sampled.data["deck"]
    assert len(all_cards) == len(set(all_cards))


# ------------------------------------------------------------------ scores / terminal

def test_get_scores_returns_floats() -> None:
    a = _adapter()
    state = _state(
        hands={0: [], 1: [], 2: []},
        books={0: [0], 1: [1, 2], 2: []},
        game_over=True,
        n_players=3,
    )
    scores = a.get_scores(state)
    assert scores == {0: 1.0, 1: 2.0, 2: 0.0}
    assert all(isinstance(v, float) for v in scores.values())


def test_is_terminal_false_initially() -> None:
    a = _adapter()
    state = a.get_initial_state()
    assert not a.is_terminal(state)


def test_is_terminal_true_when_flag_set() -> None:
    a = _adapter()
    state = a.get_initial_state()
    state.data["game_over"] = True
    assert a.is_terminal(state)


# ------------------------------------------------------------------ action label

def test_action_label_groups_by_family_and_component() -> None:
    a = _adapter()
    label1 = a.get_action_label(_ask(1, 2, 3))
    label2 = a.get_action_label(_ask(0, 2, 3))  # different target, same card
    assert label1 == label2


def test_action_label_pass() -> None:
    a = _adapter()
    assert a.get_action_label(_pass()) == "pass"


# ------------------------------------------------------------------ integration

def test_random_agents_game_terminates() -> None:
    a = _adapter(n_players=4, n_families=7, n_components=6, seed=0)
    agents = [RandomAgent(seed=i) for i in range(4)]
    engine = SimulationEngine(a, agents)
    results = engine.run_batch(n_games=20)
    assert len(results) == 20
    assert all(not r.timed_out for r in results)


def test_random_agents_all_families_booked_at_end() -> None:
    a = _adapter(n_players=3, n_families=4, n_components=4, seed=1)
    agents = [RandomAgent(seed=i) for i in range(3)]
    engine = SimulationEngine(a, agents)
    for _ in range(10):
        result = engine.run_game()
        total_books = sum(result.scores.values())
        assert total_books == 4  # all families booked


@pytest.mark.parametrize("n_players,n_families,n_components", [
    (2, 3, 3),
    (3, 7, 6),
    (5, 4, 4),
])
def test_various_configs_complete_without_error(
    n_players: int, n_families: int, n_components: int
) -> None:
    a = SevenFamiliesAdapter(n_players=n_players, n_families=n_families, n_components=n_components, seed=42)
    agents = [RandomAgent(seed=i) for i in range(n_players)]
    engine = SimulationEngine(a, agents)
    results = engine.run_batch(n_games=5)
    assert len(results) == 5
    assert all(not r.timed_out for r in results)
