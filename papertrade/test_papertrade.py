"""Checks on the parts where a quiet mistake would fake a result. Run: pytest papertrade/"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from backtest import STAKE, kalshi_fee, missing_minutes, paper_trade
from fetch import parse_ticker


def test_parse_ticker_is_away_then_home() -> None:
    assert parse_ticker("KXNBAGAME-26JUN13NYKSAS-SAS") == (
        dt.date(2026, 6, 13),
        "NYK",
        "SAS",
        "SAS",
    )
    assert parse_ticker("KXNBAGAME-26JUN13NYKSAS-NYK") == (
        dt.date(2026, 6, 13),
        "NYK",
        "SAS",
        "NYK",
    )


def test_missing_minutes_uses_only_earlier_games() -> None:
    star_plays = {"star": 36, "bench": 12}
    star_sits = {"star": 0, "bench": 30}
    shares = missing_minutes([("g1", star_plays), ("g2", star_plays), ("g3", star_sits)])
    assert shares["g1"] == 0.0  # no history yet
    assert shares["g2"] == 0.0  # everyone who usually plays, played
    assert shares["g3"] == pytest.approx(36 / 240)  # the star's usual 36 minutes are out

    # Changing how much anyone played in g3 must not change g3's own value:
    # only *who* was available for it may matter.
    altered = missing_minutes(
        [("g1", star_plays), ("g2", star_plays), ("g3", {"star": 0, "bench": 48})]
    )
    assert altered["g3"] == shares["g3"]


def test_paper_trade_pnl_and_no_trade_zone() -> None:
    bid, ask = np.array([0.58]), np.array([0.60])
    # Model agrees with the market: no edge either way, no trade.
    assert paper_trade(np.array([0.59]), bid, ask, np.array([True]))[0] == 0.0

    # Model likes the home side at 0.70: buy YES at the ask.
    fee = kalshi_fee(0.60)
    win = paper_trade(np.array([0.70]), bid, ask, np.array([True]))[0]
    loss = paper_trade(np.array([0.70]), bid, ask, np.array([False]))[0]
    assert win == pytest.approx(STAKE / 0.60 * (1 - 0.60 - fee))
    assert loss == pytest.approx(STAKE / 0.60 * (-0.60 - fee))

    # Model likes the away side: buying NO costs 1 - bid, and wins when home loses.
    away = paper_trade(np.array([0.40]), bid, ask, np.array([False]))[0]
    assert away == pytest.approx(STAKE / 0.42 * (1 - 0.42 - kalshi_fee(0.42)))
