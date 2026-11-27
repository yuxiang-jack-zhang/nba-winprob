"""Phase 3 model pieces on hand-built data: snapshots, walk-forward split, bootstrap, costs."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from nba_winprob.engine.events import Event, Settle, sort_key
from nba_winprob.ingame.model import game_ci, game_snapshots, taker_trades, walk_forward
from tests.test_run_reversal import T0, G, Game


def synthetic_game(pregame: tuple[float, float] = (0.59, 0.61)) -> list[Event]:
    g = Game(0, 0)
    q = g.quote
    events: list[Event] = [
        q(T0 - 300, 0.69, 0.71),  # pregame, superseded
        q(T0, *pregame),  # last quote at or before the first play
        g.play(0),  # tip
        q(T0 + 10, 0.60, 0.62),  # no play yet at ts - 30: not in-game
        q(T0 + 100, 0.61, 0.63),
        g.score(100, "home", 3),
        q(T0 + 200, 0.63, 0.65),
        g.score(200, "away", 2),
        g.timeout(229, "home"),  # 61 s before the lagged clock at T0+320: out of the window
        g.timeout(250, "away"),
        q(T0 + 280, 0.64, 0.66),  # within 60 s of T0+320: not its 1-minute reference
        g.score(290, "home", 2),
        g.timeout(300, "home"),
        g.score(305, "home", 3),
        q(T0 + 320, 0.65, 0.67),  # sees plays up to ts T0+290 only
        q(T0 + 335, 0.66, 0.68),  # sees the play at exactly ts - 30
        q(T0 + 500, 0.69, 0.71),
        q(T0 + 520, 0.79, 0.81),
        q(T0 + 600, 0.96, 0.98),  # mid 0.97: out of range (strict)
        q(T0 + 700, 0.02, 0.04),  # mid 0.03: out of range (strict)
        g.play(2755),
        q(T0 + 2790, 0.50, 0.52),  # 125 s left at the lagged play
        g.play(2765),
        q(T0 + 2795, 0.50, 0.52),  # 115 s left: filtered
        Settle(ts=T0 + 2880, game_id=G, home_won=True),
    ]
    return sorted(events, key=sort_key)


@pytest.fixture(scope="module")
def snaps() -> pd.DataFrame:
    return game_snapshots(synthetic_game()).set_index("ts")


def test_filters_keep_only_in_game_quotes_in_range_with_over_two_minutes_left(
    snaps: pd.DataFrame,
) -> None:
    kept = [T0 + s for s in (100, 200, 280, 320, 335, 500, 520, 2790)]
    assert snaps.index.tolist() == kept


def test_features_use_only_plays_at_least_30s_before_the_quote(snaps: pd.DataFrame) -> None:
    before, at = snaps.loc[T0 + 320], snaps.loc[T0 + 335]
    assert (before["margin"], before["home_timeout"]) == (3, False)
    assert (at["margin"], at["home_timeout"]) == (6, True)


def test_feature_values(snaps: pd.DataFrame) -> None:
    row = snaps.loc[T0 + 320]
    assert row["seconds_left"] == 2880 - 290
    assert row["period"] == 1
    assert row["margin_scaled"] == pytest.approx(3 / math.sqrt(2590 + 60))
    assert row["run_240"] == 3  # window (50, 290]: home 3 + 2, away 2
    assert row["run_120"] == 0  # window (170, 290]: home 2, away 2
    assert (row["home_timeout"], row["away_timeout"]) == (False, True)
    assert row["logit_mid"] == pytest.approx(math.log(0.66 / 0.34))
    assert row["pregame_logit"] == pytest.approx(math.log(0.6 / 0.4))
    assert row["dmid_1m"] == pytest.approx(0.66 - 0.64)  # last quote <= ts - 60: T0+200
    assert row["dmid_3m"] == pytest.approx(0.66 - 0.62)  # last quote <= ts - 180: T0+100
    assert row["move"] == pytest.approx(0.70 - 0.66)  # first quote >= ts + 180: T0+500
    assert row["home_won"]
    assert math.isnan(snaps.loc[T0 + 2790, "move"])  # no quote 3 minutes later


def test_pregame_mid_is_clipped_before_the_logit() -> None:
    snaps = game_snapshots(synthetic_game(pregame=(0.995, 0.999)))
    assert snaps["pregame_logit"].iloc[0] == pytest.approx(math.log(0.99 / 0.01))


def test_walk_forward_never_trains_on_a_game_dated_at_or_after_a_test_game() -> None:
    # 7 days x 3 games, shuffled; the 40/55/70/85% cuts land mid-day and move to day start.
    games = pd.DataFrame(
        {
            "game_id": [f"g{i:02d}" for i in range(21)],
            "day": [f"2026-01-0{i // 3 + 1}" for i in range(21)],
            "tip_ts": range(21),
        }
    ).sample(frac=1, random_state=1)
    day = games.set_index("game_id")["day"]
    blocks = walk_forward(games)
    assert [(len(tr), len(te)) for tr, te in blocks] == [(6, 3), (9, 3), (12, 3), (15, 6)]
    for train, test in blocks:
        assert day[train].max() < day[test].min()
        assert set(train) == {g for g in day.index if day[g] < day[test].min()}


def test_bootstrap_resamples_games_not_snapshots() -> None:
    one_game = pd.DataFrame({"game_id": "a", "x": range(100), "one": 1.0})
    lo, hi = game_ci(one_game, "x", "one", pd.Index(["a"]))
    assert lo == hi == pytest.approx(49.5)  # one game: every resample is the same

    two_games = pd.DataFrame({"game_id": ["a"] * 50 + ["b"] * 50, "x": [1.0] * 50 + [0.0] * 50})
    two_games["one"] = 1.0
    assert game_ci(two_games, "x", "one", pd.Index(["a", "b"])) == (0.0, 1.0)


def test_taker_trades_cost_and_direction() -> None:
    preds = pd.DataFrame(
        {
            "game_id": ["a", "b", "c", "d"],
            "bid": [0.49, 0.49, 0.30, 0.49],
            "ask": [0.51, 0.51, 0.34, 0.51],
            "B1": [0.05, 0.06, -0.08, 0.5],
            "move": [0.06, 0.06, -0.02, math.nan],
        }
    )
    cost_b = 0.02 + 2 * 0.07 * 0.51 * 0.49  # spread + fee on both legs at the ask
    cost_c = 0.04 + 2 * 0.07 * 0.70 * 0.30  # buying NO at 1 - bid = 0.70
    trades = taker_trades(preds, "B1")
    assert trades["game_id"].tolist() == ["b", "c"]
    assert trades["net"].tolist() == pytest.approx([0.06 - cost_b, 0.02 - cost_c])
