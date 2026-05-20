from __future__ import annotations

from core.engine import SimulationEngine
from core.types import Action, State
from games.odin.adapter import OdinAdapter, _homogeneous, _value
from rl.random_agent import RandomAgent


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_state(
    hands: dict[int, list],
    table: list,
    current: int = 0,
    round_starter: int = 0,
    last_placer: int | None = None,
    consecutive_passes: int = 0,
) -> State:
    n = len(hands)
    return State(
        data={
            "hands": hands,
            "scores": {pid: 0 for pid in range(n)},
            "table": table,
            "round_starter": round_starter,
            "last_placer": last_placer,
            "hand_starter": round_starter,
            "current": current,
            "consecutive_passes": consecutive_passes,
            "reserve": [],
            "n_players": n,
            "game_over": False,
            "points_limit": 15,
        }
    )


# ── _value & _homogeneous ─────────────────────────────────────────────────────


def test_value_single_card() -> None:
    assert _value([(7, 0)]) == 7


def test_value_two_cards_sorted_descending() -> None:
    # The highest arrangement of 2 and 8 is 82, not 28.
    assert _value([(2, 0), (8, 1)]) == 82


def test_value_three_cards() -> None:
    assert _value([(2, 0), (4, 1), (9, 2)]) == 942


def test_homogeneous_single_card() -> None:
    assert _homogeneous([(5, 0)])


def test_homogeneous_same_number() -> None:
    assert _homogeneous([(3, 0), (3, 1), (3, 2)])


def test_homogeneous_same_colour() -> None:
    assert _homogeneous([(1, 2), (5, 2), (9, 2)])


def test_homogeneous_mixed_fails() -> None:
    assert not _homogeneous([(1, 0), (2, 1)])


def test_homogeneous_mixed_three_fails() -> None:
    assert not _homogeneous([(1, 0), (1, 1), (2, 0)])


# ── round start (table empty) ─────────────────────────────────────────────────


def test_round_start_no_pass_action() -> None:
    """The round-starter cannot pass — they must open with a card."""
    adapter = OdinAdapter(n_players=2, seed=0)
    state = _make_state(
        hands={0: [(1, 0), (2, 1), (3, 2)], 1: [(4, 0)]},
        table=[],
    )
    legal = adapter.get_legal_actions(state, 0)
    assert legal, "must have at least one legal action"
    assert not any(a.data["type"] == "pass" for a in legal)


def test_round_start_each_action_plays_exactly_one_card() -> None:
    """A non-homogeneous hand at round start only produces 1-card plays."""
    adapter = OdinAdapter(n_players=2, seed=0)
    state = _make_state(
        hands={0: [(1, 0), (3, 1), (5, 2), (7, 3)], 1: [(2, 0)]},
        table=[],
    )
    legal = adapter.get_legal_actions(state, 0)
    for a in legal:
        assert len(a.data["cards"]) == 1


def test_round_start_homogeneous_hand_adds_play_all() -> None:
    """With a fully homogeneous hand, a play-all action must be offered."""
    adapter = OdinAdapter(n_players=2, seed=0)
    hand = [(1, 0), (4, 0), (7, 0)]
    state = _make_state(hands={0: hand, 1: [(2, 1)]}, table=[])
    legal = adapter.get_legal_actions(state, 0)
    play_all = [a for a in legal if len(a.data["cards"]) == 3]
    assert len(play_all) == 1
    assert sorted(play_all[0].data["cards"]) == sorted(hand)


def test_round_start_one_card_hand_no_duplicate() -> None:
    """A 1-card hand is trivially homogeneous but must not produce duplicates."""
    adapter = OdinAdapter(n_players=2, seed=0)
    state = _make_state(hands={0: [(5, 0)], 1: [(2, 1)]}, table=[])
    legal = adapter.get_legal_actions(state, 0)
    assert len(legal) == 1
    assert legal[0].data["cards"] == [(5, 0)]


# ── action legality: count constraint ─────────────────────────────────────────


def test_count_must_be_same_or_one_more_than_table() -> None:
    """With 2 cards on the table, only plays of size 2 or 3 are legal."""
    adapter = OdinAdapter(n_players=2, seed=0)
    state = _make_state(
        hands={0: [(9, 0), (8, 0), (7, 0), (6, 0)], 1: [(1, 1)]},
        table=[(3, 1), (2, 1)],  # table value = 32
        current=0,
        round_starter=1,
        last_placer=1,
    )
    legal = adapter.get_legal_actions(state, 0)
    for a in legal:
        if a.data["type"] == "play":
            assert len(a.data["cards"]) in (2, 3), (
                f"play of size {len(a.data['cards'])} illegal with 2 cards on table"
            )


def test_cannot_play_single_card_on_two_card_table() -> None:
    adapter = OdinAdapter(n_players=2, seed=0)
    state = _make_state(
        hands={0: [(9, 0), (8, 1), (7, 2)], 1: [(1, 0)]},
        table=[(3, 1), (2, 1)],
        current=0,
        round_starter=1,
        last_placer=1,
    )
    legal = adapter.get_legal_actions(state, 0)
    assert not any(
        a.data["type"] == "play" and len(a.data["cards"]) == 1 for a in legal
    )


def test_cannot_play_when_hand_too_small_for_table() -> None:
    """Player with fewer cards than required can only pass."""
    adapter = OdinAdapter(n_players=2, seed=0)
    state = _make_state(
        hands={0: [(9, 0), (8, 1)], 1: [(1, 0)]},  # only 2 cards
        table=[(5, 0), (4, 1), (3, 2)],  # table has 3 → need 3 or 4
        current=0,
        round_starter=1,
        last_placer=1,
    )
    legal = adapter.get_legal_actions(state, 0)
    assert legal == [Action(data={"type": "pass"})]


# ── action legality: value constraint ─────────────────────────────────────────


def test_play_value_must_be_strictly_greater() -> None:
    """Every generated play action must beat the current table value."""
    adapter = OdinAdapter(n_players=2, seed=0)
    state = _make_state(
        hands={0: [(3, 0), (5, 1), (7, 2), (9, 3), (6, 0)], 1: [(1, 1)]},
        table=[(6, 1)],  # table value = 6
        current=0,
        round_starter=1,
        last_placer=1,
    )
    legal = adapter.get_legal_actions(state, 0)
    for a in legal:
        if a.data["type"] == "play":
            assert _value(a.data["cards"]) > 6, (
                f"play value {_value(a.data['cards'])} must exceed 6"
            )


def test_no_plays_when_nothing_beats_table() -> None:
    """When no set in hand can beat the table, only pass is legal."""
    adapter = OdinAdapter(n_players=2, seed=0)
    # Table has 3 cards of the highest values; hand cannot match count or value.
    state = _make_state(
        hands={0: [(1, 0), (2, 1)], 1: [(1, 1)]},  # 2 cards only
        table=[(9, 0), (8, 0), (7, 0)],  # need 3 or 4; player has only 2
        current=0,
        round_starter=1,
        last_placer=1,
    )
    legal = adapter.get_legal_actions(state, 0)
    assert legal == [Action(data={"type": "pass"})]


# ── action legality: homogeneity constraint ───────────────────────────────────


def test_multi_card_plays_must_be_homogeneous() -> None:
    """Every multi-card play in the legal list must share a number or colour."""
    adapter = OdinAdapter(n_players=2, seed=0)
    state = _make_state(
        hands={0: [(9, 0), (8, 1), (7, 2), (6, 3)], 1: [(1, 0)]},
        table=[(5, 0), (4, 0)],  # table value = 54
        current=0,
        round_starter=1,
        last_placer=1,
    )
    legal = adapter.get_legal_actions(state, 0)
    for a in legal:
        if a.data["type"] == "play":
            assert _homogeneous(a.data["cards"]), (
                f"non-homogeneous set in legal action: {a.data['cards']}"
            )


# ── pass availability ─────────────────────────────────────────────────────────


def test_pass_always_available_when_table_nonempty() -> None:
    adapter = OdinAdapter(n_players=2, seed=0)
    state = _make_state(
        hands={0: [(8, 0), (7, 0), (6, 0)], 1: [(1, 1)]},
        table=[(5, 1)],
        current=0,
        round_starter=1,
        last_placer=1,
    )
    legal = adapter.get_legal_actions(state, 0)
    assert any(a.data["type"] == "pass" for a in legal)


def test_pass_available_even_when_valid_plays_exist() -> None:
    """Pass is legal even when plays are available — the player may choose to
    avoid a pickup that weakens their hand."""
    adapter = OdinAdapter(n_players=2, seed=0)
    # Hand is monochromatic (colour 0); the only card to pick up is colour 1,
    # which would break that cohesion.
    state = _make_state(
        hands={0: [(8, 0), (7, 0), (6, 0)], 1: [(1, 0)]},
        table=[(4, 1)],  # only pickup is (4, colour-1)
        current=0,
        round_starter=1,
        last_placer=1,
    )
    legal = adapter.get_legal_actions(state, 0)
    play_actions = [a for a in legal if a.data["type"] == "play"]
    assert play_actions, "values 8, 7, 6 all beat 4 — plays must exist"
    assert any(a.data["type"] == "pass" for a in legal), (
        "pass must be legal so the player can decline the colour-mismatched pickup"
    )


def test_all_pickups_are_from_old_table() -> None:
    """The pickup in every play action belongs to the current table, not the
    newly played cards (rule: 'you cannot pick up a card you just played')."""
    adapter = OdinAdapter(n_players=2, seed=0)
    table = [(4, 1), (3, 1)]  # two colour-1 cards on the table
    state = _make_state(
        hands={0: [(9, 0), (8, 0), (7, 0)], 1: [(1, 0)]},
        table=table,
        current=0,
        round_starter=1,
        last_placer=1,
    )
    legal = adapter.get_legal_actions(state, 0)
    table_cards = {tuple(c) for c in table}
    for a in legal:
        if a.data["type"] == "play" and a.data["pickup"] is not None:
            assert tuple(a.data["pickup"]) in table_cards, (
                f"pickup {a.data['pickup']} not in old table {table}"
            )


# ── no pickup when hand empties ───────────────────────────────────────────────


def test_no_pickup_action_when_play_empties_hand() -> None:
    """Playing the last card(s) must produce a pickup=None action."""
    adapter = OdinAdapter(n_players=2, seed=0)
    state = _make_state(
        hands={0: [(8, 0)], 1: [(1, 1)]},  # one card left
        table=[(5, 1)],
        current=0,
        round_starter=1,
        last_placer=1,
    )
    legal = adapter.get_legal_actions(state, 0)
    play = next(a for a in legal if a.data["type"] == "play" and a.data["cards"] == [(8, 0)])
    assert play.data["pickup"] is None


# ── apply_action ──────────────────────────────────────────────────────────────


def test_play_removes_card_and_sets_table() -> None:
    adapter = OdinAdapter(n_players=2, seed=0)
    state = _make_state(
        hands={0: [(8, 0), (7, 0), (3, 1)], 1: [(1, 1)]},
        table=[(5, 1)],
        current=0,
        round_starter=1,
        last_placer=1,
    )
    action = Action(data={"type": "play", "cards": [(8, 0)], "pickup": (5, 1)})
    new = adapter.apply_action(state, action, 0)
    assert (8, 0) not in new.data["hands"][0]
    assert new.data["table"] == [(8, 0)]
    assert (5, 1) in new.data["hands"][0]


def test_pass_increments_counter_and_resets_round_at_n_minus_1() -> None:
    """With 2 players, a single pass is enough (n−1 = 1) to end the round."""
    adapter = OdinAdapter(n_players=2, seed=0)
    state = _make_state(
        hands={0: [(1, 0)], 1: [(5, 1)]},
        table=[(9, 0)],
        current=0,
        round_starter=1,
        last_placer=1,
        consecutive_passes=0,
    )
    new = adapter.apply_action(state, Action(data={"type": "pass"}), 0)
    assert new.data["table"] == []
    assert new.data["consecutive_passes"] == 0
    assert new.data["round_starter"] == 1  # last_placer
    assert new.data["current"] == 1


def test_round_ends_after_n_minus_1_passes_three_players() -> None:
    """With 3 players, exactly 2 consecutive passes must end the round."""
    adapter = OdinAdapter(n_players=3, seed=0)
    state = _make_state(
        hands={0: [(1, 0)], 1: [(2, 1)], 2: [(3, 2)]},
        table=[(9, 0)],
        current=1,
        round_starter=0,
        last_placer=0,
        consecutive_passes=0,
    )
    s1 = adapter.apply_action(state, Action(data={"type": "pass"}), 1)
    assert s1.data["table"] != [], "round must still be open after 1 pass"
    assert s1.data["consecutive_passes"] == 1

    s2 = adapter.apply_action(s1, Action(data={"type": "pass"}), 2)
    assert s2.data["table"] == [], "round must end after 2nd consecutive pass"
    assert s2.data["consecutive_passes"] == 0
    assert s2.data["round_starter"] == 0  # last_placer was player 0


def test_play_resets_consecutive_passes() -> None:
    """A play after one pass must reset the consecutive-pass counter."""
    adapter = OdinAdapter(n_players=3, seed=0)
    state = _make_state(
        hands={0: [(9, 0)], 1: [(2, 1)], 2: [(3, 1), (4, 1)]},
        table=[(5, 0)],
        current=2,
        round_starter=0,
        last_placer=0,
        consecutive_passes=1,  # player 1 already passed
    )
    action = Action(data={"type": "play", "cards": [(4, 1)], "pickup": (5, 0)})
    new = adapter.apply_action(state, action, 2)
    assert new.data["consecutive_passes"] == 0


def test_empty_hand_ends_hand_and_scores_remaining() -> None:
    """Playing the last card ends the hand; opponent's remaining cards score."""
    adapter = OdinAdapter(n_players=2, seed=0)
    state = _make_state(
        hands={0: [(8, 0)], 1: [(3, 1), (2, 1)]},
        table=[(5, 1)],
        current=0,
        round_starter=1,
        last_placer=1,
    )
    action = Action(data={"type": "play", "cards": [(8, 0)], "pickup": None})
    new = adapter.apply_action(state, action, 0)
    # Whether game_over or a new hand started, scores must reflect the hand result.
    assert new.data["scores"][0] == 0, "winner gets 0 penalty points"
    assert new.data["scores"][1] == 2, "loser had 2 cards remaining"


def test_state_mutation_is_isolated() -> None:
    """apply_action must not mutate the original state."""
    adapter = OdinAdapter(n_players=2, seed=0)
    state = _make_state(
        hands={0: [(8, 0), (7, 0)], 1: [(1, 1)]},
        table=[(5, 1)],
        current=0,
        round_starter=1,
        last_placer=1,
    )
    original_hand = list(state.data["hands"][0])
    original_table = list(state.data["table"])
    action = Action(data={"type": "play", "cards": [(8, 0)], "pickup": (5, 1)})
    adapter.apply_action(state, action, 0)
    assert state.data["hands"][0] == original_hand
    assert state.data["table"] == original_table


# ── observable state ──────────────────────────────────────────────────────────


def test_observable_state_hides_opponent_hand() -> None:
    adapter = OdinAdapter(n_players=2, seed=0)
    state = _make_state(
        hands={0: [(1, 0), (2, 0)], 1: [(9, 1), (8, 1), (7, 1)]},
        table=[],
    )
    obs = adapter.get_observable_state(state, 0)
    assert obs.data["hands"][0] == state.data["hands"][0], "own hand must be visible"
    assert obs.data["hands"][1] == [], "opponent hand must be hidden (empty)"
    assert obs.data["hand_sizes"][1] == 3, "opponent size must still be known"


# ── full-game integration ─────────────────────────────────────────────────────


def test_game_runs_to_completion_two_players() -> None:
    adapter = OdinAdapter(n_players=2, seed=42)
    agents = [RandomAgent(seed=i) for i in range(2)]
    result = SimulationEngine(adapter, agents, max_turns=100_000).run_game()
    assert not result.timed_out
    assert result.n_turns > 0


def test_game_runs_to_completion_four_players() -> None:
    adapter = OdinAdapter(n_players=4, seed=42)
    agents = [RandomAgent(seed=i) for i in range(4)]
    result = SimulationEngine(adapter, agents, max_turns=100_000).run_game()
    assert not result.timed_out
    assert result.n_turns > 0


def test_no_illegal_actions_across_multiple_games() -> None:
    """The engine should never have to fall back to a random replacement."""
    adapter = OdinAdapter(n_players=3, seed=7)
    agents = [RandomAgent(seed=i) for i in range(3)]
    engine = SimulationEngine(adapter, agents, max_turns=100_000)
    for game_idx in range(20):
        result = engine.run_game()
        total_illegal = sum(result.illegal_action_counts.values())
        assert total_illegal == 0, f"game {game_idx}: {result.illegal_action_counts}"


def test_scores_are_non_positive() -> None:
    """Scores are negated penalties, so the final values must all be ≤ 0."""
    adapter = OdinAdapter(n_players=2, seed=1)
    agents = [RandomAgent(seed=i) for i in range(2)]
    engine = SimulationEngine(adapter, agents)
    for _ in range(30):
        result = engine.run_game()
        assert all(s <= 0 for s in result.scores.values())


def test_random_agents_play_cards_not_only_pass() -> None:
    """Over many turns, random agents must sometimes play cards (not only pass).
    This guards against a bug where get_legal_actions only ever returns pass."""
    adapter = OdinAdapter(n_players=3, seed=0)
    agents = [RandomAgent(seed=i) for i in range(3)]
    engine = SimulationEngine(adapter, agents, max_turns=100_000, record=True)
    result = engine.run_game()
    assert result.actions is not None
    play_count = sum(1 for _, _, a in result.actions if a.data.get("type") == "play")
    pass_count = sum(1 for _, _, a in result.actions if a.data.get("type") == "pass")
    assert play_count > 0, "no play actions recorded — adapter may always return only pass"
    # Passes should not represent more than 80 % of all actions in a healthy game.
    ratio = pass_count / (play_count + pass_count)
    assert ratio < 0.80, f"pass ratio {ratio:.0%} is suspiciously high"
