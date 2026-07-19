from __future__ import annotations

from core.match_session import MatchSession


# -------- rotation --------------------------------------------------------


def test_rotate_two_players_cycles_every_round() -> None:
    session = MatchSession(n_players=2, human_seat=0)
    seats = ["H", "B"]

    assert session.rotate(seats) == ["H", "B"]
    session.next_round()
    assert session.rotate(seats) == ["B", "H"]
    session.next_round()
    assert session.rotate(seats) == ["H", "B"]


def test_rotate_three_players_cycles_every_n_rounds() -> None:
    session = MatchSession(n_players=3, human_seat=1)
    seats = ["A", "B", "C"]

    expected = [
        ["A", "B", "C"],
        ["B", "C", "A"],
        ["C", "A", "B"],
        ["A", "B", "C"],
    ]
    for want in expected:
        assert session.rotate(seats) == want
        session.next_round()


# -------- human_player_id --------------------------------------------------


def test_human_player_id_tracks_rotation_n2() -> None:
    session = MatchSession(n_players=2, human_seat=0)
    seats = ["H", "B"]

    for _ in range(4):
        rotated = session.rotate(seats)
        assert rotated[session.human_player_id()] == "H"
        session.next_round()


def test_human_player_id_tracks_rotation_n3() -> None:
    session = MatchSession(n_players=3, human_seat=1)
    seats = ["A", "B", "C"]

    for _ in range(5):
        rotated = session.rotate(seats)
        assert rotated[session.human_player_id()] == "B"
        session.next_round()


# -------- record ------------------------------------------------------------


def test_record_attributes_win_to_original_seat_n2() -> None:
    session = MatchSession(n_players=2, human_seat=0)
    session.next_round()  # round 2: agents = [B, H], offset = 1

    session.record(winner_id=1)  # H wins as pid 1 -> original seat 0

    assert session.wins == {0: 1, 1: 0}


def test_record_attributes_win_to_original_seat_n3() -> None:
    session = MatchSession(n_players=3, human_seat=1)
    session.next_round()  # round 2: agents = [B, C, A], offset = 1

    session.record(winner_id=0)  # B wins as pid 0 -> original seat 1
    assert session.wins[1] == 1

    session.record(winner_id=2)  # A wins as pid 2 -> original seat 0
    assert session.wins[0] == 1
    assert session.wins == {0: 1, 1: 1, 2: 0}


def test_record_draw_increments_draws_not_wins() -> None:
    session = MatchSession(n_players=2, human_seat=0)

    session.record(winner_id=None)

    assert session.draws == 1
    assert session.wins == {0: 0, 1: 0}


# -------- reset --------------------------------------------------------------


def test_reset_returns_to_round_one_and_zeroes_tally() -> None:
    session = MatchSession(n_players=2, human_seat=0)
    session.record(winner_id=0)
    session.next_round()
    session.record(winner_id=None)

    session.reset()

    assert session.round_number == 1
    assert session.wins == {0: 0, 1: 0}
    assert session.draws == 0


def test_wins_prepopulated_for_every_seat() -> None:
    session = MatchSession(n_players=3, human_seat=2)

    assert session.wins == {0: 0, 1: 0, 2: 0}
