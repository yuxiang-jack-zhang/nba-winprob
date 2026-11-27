"""Game clock and trailing-run tracking on synthetic play sequences."""

from __future__ import annotations

import pytest

from nba_winprob.engine.events import Play, PlayKind, Side
from nba_winprob.features.ingame import RunTracker, game_seconds_elapsed, game_seconds_left


def play(
    e: float,
    kind: PlayKind = "other",
    team: Side | None = None,
    points: int = 0,
    home: int = 0,
    away: int = 0,
    ts: float | None = None,
) -> Play:
    """A regulation play at ``e`` game seconds elapsed."""
    period = min(int(e // 720) + 1, 4)
    clock = period * 720 - e
    return Play(
        ts=1000 + e if ts is None else ts,
        game_id="g",
        seq=0,
        period=period,
        clock=clock,
        home_score=home,
        away_score=away,
        team=team,
        kind=kind,
        points=points,
        text="",
    )


@pytest.mark.parametrize(
    ("period", "clock", "elapsed", "left"),
    [
        (1, 720, 0, 2880),
        (1, 0, 720, 2160),
        (3, 451, 1709, 1171),
        (4, 120, 2760, 120),
        (4, 0, 2880, 0),
        (5, 300, 2880, 300),  # OT starts where regulation ended
        (5, 121, 3059, 121),  # "left" in OT is left in this OT period
        (6, 150, 3330, 150),
    ],
)
def test_game_clock(period: int, clock: float, elapsed: float, left: float) -> None:
    assert game_seconds_elapsed(period, clock) == elapsed
    assert game_seconds_left(period, clock) == left


def test_run_window_is_open_at_the_start_and_closed_at_the_end() -> None:
    t = RunTracker(window_s=240)
    t.update(play(100, "score", "away", 3, away=3))
    t.update(play(339.5, "other", away=3))
    assert t.points("away") == 3  # 239.5 s ago: inside
    t.update(play(340, "other", away=3))
    assert t.points("away") == 0  # exactly 240 s ago: out


def test_basket_at_the_latest_clock_counts() -> None:
    t = RunTracker(window_s=240)
    t.update(play(500, "score", "home", 2, home=2))
    assert t.points("home") == 2


def test_run_net_points_and_margin() -> None:
    t = RunTracker(window_s=240)
    t.update(play(10, "score", "home", 2, home=52, away=50))  # slides out below
    t.update(play(300, "score", "away", 3, home=52, away=53))
    t.update(play(320, "score", "home", 2, home=54, away=53))
    t.update(play(330, "score", "away", 2, home=54, away=55))
    assert (t.points("home"), t.points("away")) == (2, 5)
    assert (t.run_net("home"), t.run_net("away")) == (-3, 3)
    assert t.margin == -1
    assert t.seconds_left == 2880 - 330


def test_timeout_windows_are_open_at_the_start_and_closed_at_the_end() -> None:
    t = RunTracker(window_s=240)
    t.update(play(100, "timeout", "home", ts=5000))
    t.update(play(200, "timeout", None, ts=6000))  # official timeout: nobody's
    assert t.timeout_called("home", game_window=(99, 100))
    assert not t.timeout_called("home", game_window=(100, 200))
    assert t.timeout_called("home", ts_window=(4999, 5000))
    assert not t.timeout_called("home", ts_window=(5000, 6000))
    assert not t.timeout_called("away", game_window=(0, 300), ts_window=(0, 9999))
    assert not t.timeout_called("home")


def test_timeout_either_window_suffices() -> None:
    t = RunTracker(window_s=240)
    t.update(play(100, "timeout", "away", ts=5000))
    assert t.timeout_called("away", game_window=(200, 300), ts_window=(4990, 5000))
    assert t.timeout_called("away", game_window=(50, 100), ts_window=(0, 1))
