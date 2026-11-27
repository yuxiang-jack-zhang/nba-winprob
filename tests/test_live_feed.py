"""Live sources: scoreboard, ticker mapping, top of book, and ``LiveFeed`` polling.

No network. ``espn_scoreboard.json`` is trimmed from real scoreboards: three finished
games of 2026-11-25 and one scheduled game of 2026-11-27 (MIA at NY). The Kalshi market
is that game's real ``GET /markets/{ticker}`` response, taken before tip. Feed tests
replay the trimmed game 401811001 (see test_ingame_sources.py) a few plays at a time.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nba_winprob.config import Settings
from nba_winprob.engine.events import Play, Quote, Settle
from nba_winprob.sources.espn import Game, parse_plays, parse_scoreboard
from nba_winprob.sources.kalshi import kalshi_ticker, parse_market
from nba_winprob.sources.live import LiveFeed

FIXTURES = Path(__file__).parent / "fixtures"
GAME_ID = "401811001"
TICKER = "KXNBAGAME-26APR06PHISAS-SAS"


def load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text())
    return data


@pytest.fixture
def summary() -> dict[str, Any]:
    """The trimmed game with plays in wallclock order, so ``upto`` takes a real prefix."""
    data = load(f"espn_{GAME_ID}.json")
    data["plays"].sort(key=lambda p: (p["wallclock"], int(p["sequenceNumber"])))
    return data


def upto(summary: dict[str, Any], n: int) -> dict[str, Any]:
    """The game as ESPN serves it mid-game: the first ``n`` plays, header score unchanged."""
    return {**summary, "plays": summary["plays"][:n]}


# ----------------------------------------------------------------------------- scoreboard


def test_parse_scoreboard() -> None:
    games = parse_scoreboard(load("espn_scoreboard.json"))
    assert [(g.game_id, g.away, g.home, g.state, g.home_won) for g in games] == [
        ("401909317", "MIA", "PHI", "post", True),
        ("401909318", "CHI", "CHA", "post", False),
        ("401909319", "CLE", "NY", "post", True),
        ("401909326", "MIA", "NY", "pre", None),
    ]
    assert games[3].tip_utc == "2026-11-28T00:00Z"


def test_parse_scoreboard_season_types_and_postponed() -> None:
    response = load("espn_scoreboard.json")
    for event, season_type in zip(response["events"], (1, 3, 5, 4), strict=True):
        event["season"]["type"] = season_type  # preseason, playoffs, play-in, All-Star
    status = response["events"][1]["competitions"][0]["status"]["type"]
    status.update(name="STATUS_POSTPONED", completed=False)
    assert [(g.game_id, g.home_won) for g in parse_scoreboard(response)] == [
        ("401909318", None),
        ("401909319", True),
    ]


# ----------------------------------------------------------------------------- tickers


def test_kalshi_ticker_fixtures() -> None:
    game = parse_scoreboard(load("espn_scoreboard.json"))[3]
    # Tips 00:00 UTC on the 28th, which is the 27th in New York; NY is NYK on Kalshi.
    expected = load("kalshi_market_KXNBAGAME-26NOV27MIANYK-NYK.json")["market"]["ticker"]
    assert kalshi_ticker(game.tip_utc, game.away, game.home) == expected
    assert kalshi_ticker("2026-04-07T00:00Z", "PHI", "SA") == TICKER


def test_kalshi_ticker_matches_every_real_market(settings: Settings) -> None:
    root = settings.data_dir / "papertrade"
    if not (root / "kalshi.parquet").exists() or not (root / "games.parquet").exists():
        pytest.skip("data/papertrade is absent; run papertrade/fetch.py")
    markets = pd.read_parquet(root / "kalshi.parquet")
    games = pd.read_parquet(root / "games.parquet")
    rows = markets[["game_id", "ticker"]].merge(games, on="game_id")
    assert len(rows) == len(markets) > 1000
    computed = [kalshi_ticker(r.tip_utc, r.away, r.home) for r in rows.itertuples()]
    assert computed == rows["ticker"].tolist()


# ----------------------------------------------------------------------------- quotes


def test_parse_market() -> None:
    response = load("kalshi_market_KXNBAGAME-26NOV27MIANYK-NYK.json")
    assert parse_market(response, "401909326", 5.0) == Quote(
        ts=5.0,
        game_id="401909326",
        ticker="KXNBAGAME-26NOV27MIANYK-NYK",
        yes_bid=0.60,
        yes_ask=0.62,
        volume=0,
    )


@pytest.mark.parametrize(
    ("bid", "ask"), [("0.0000", "1.0000"), ("0.0000", "0.6200"), ("0.6000", "1.0000")]
)
def test_parse_market_one_sided_or_empty_book_is_none(bid: str, ask: str) -> None:
    response = load("kalshi_market_KXNBAGAME-26NOV27MIANYK-NYK.json")
    response["market"].update(yes_bid_dollars=bid, yes_ask_dollars=ask)
    assert parse_market(response, "401909326", 5.0) is None


# ----------------------------------------------------------------------------- feed


def test_live_parse_tolerates_header_ahead_of_plays(summary: dict[str, Any]) -> None:
    """Mid-game the header already shows a later score; only history enforces it."""
    with pytest.raises(ValueError, match="ESPN final"):
        parse_plays(upto(summary, 10), GAME_ID)
    assert len(parse_plays(upto(summary, 10), GAME_ID, check_final=False)) == 10


class Fake:
    """Scripted ESPN and Kalshi: tests set ``games`` and ``summary`` between polls."""

    def __init__(self, summary: dict[str, Any] | None) -> None:
        self.games = [Game(GAME_ID, "2026-04-07T00:00Z", "SA", "PHI", "in", None)]
        self.summary = summary  # None: ESPN is down
        self.calls: list[str] = []

    def scoreboard(self, day: date) -> list[Game]:
        self.calls.append("scoreboard")
        return self.games

    def get_summary(self, game_id: str) -> dict[str, Any]:
        self.calls.append("summary")
        if self.summary is None:
            raise OSError("ESPN down")
        return self.summary

    def quote(self, ticker: str, game_id: str, ts: float) -> Quote | None:
        self.calls.append(f"quote {ticker}")
        return Quote(ts, game_id, ticker, 0.40, 0.42, 0)

    def feed(self) -> LiveFeed:
        return LiveFeed(self.scoreboard, self.get_summary, self.quote)


def test_plays_stamped_on_receipt_in_replay_order_then_quote(summary: dict[str, Any]) -> None:
    fake = Fake(upto(summary, 12))
    feed = fake.feed()
    events = feed.poll(1000.0)
    expected = parse_plays(upto(summary, 12), GAME_ID, check_final=False)
    assert [e.seq for e in events if isinstance(e, Play)] == [p.seq for p in expected]
    assert all(e.ts == 1000.0 for e in events)
    assert len(events) == 12  # the backlog, with no quote to trade it on
    fake.summary = upto(summary, 13)
    events = feed.poll(1005.0)
    assert [type(e) for e in events] == [Play, Quote]  # new plays first, then the quote
    assert events[-1] == Quote(1005.0, GAME_ID, TICKER, 0.40, 0.42, 0)


def test_plays_are_emitted_once_across_polls(summary: dict[str, Any]) -> None:
    fake = Fake(upto(summary, 10))
    feed = fake.feed()
    assert len(feed.poll(0.0)) == 10  # backlog only
    assert [type(e) for e in feed.poll(5.0)] == [Quote]
    fake.summary = upto(summary, 15)
    events = feed.poll(10.0)
    assert [e.seq for e in events if isinstance(e, Play)] == [
        int(p["sequenceNumber"]) for p in summary["plays"][10:15]
    ]


def test_late_inserted_play_gets_rebuilt_score_and_replay_order(summary: dict[str, Any]) -> None:
    plays = summary["plays"]
    late = next(i for i, p in enumerate(plays) if p["scoringPlay"] and i > 5)
    plays[late]["sequenceNumber"] = "999"  # entered late: ESPN gives it a high number
    fake = Fake({**summary, "plays": plays[:late] + plays[late + 1 : late + 4]})
    feed = fake.feed()
    feed.poll(0.0)
    fake.summary = upto(summary, late + 8)
    new = [e for e in feed.poll(5.0) if isinstance(e, Play)]

    # The late play comes first (earliest wallclock) despite its sequence number.
    assert [p.seq for p in new] == [999] + [
        int(p["sequenceNumber"]) for p in plays[late + 4 : late + 8]
    ]
    full = {p.seq: p for p in parse_plays(upto(summary, late + 8), GAME_ID, check_final=False)}
    assert [(p.home_score, p.away_score) for p in new] == [
        (full[p.seq].home_score, full[p.seq].away_score) for p in new
    ]
    assert all(p.ts == 5.0 for p in new)


def test_settles_once_when_final_then_stops_polling(summary: dict[str, Any]) -> None:
    fake = Fake(upto(summary, 10))
    feed = fake.feed()
    feed.poll(0.0)
    fake.games = [replace(fake.games[0], state="post", home_won=True)]
    fake.summary = upto(summary, 12)
    events = feed.poll(60.0)
    # The last plays still arrive, no quote after the final, then the settlement.
    assert [type(e) for e in events] == [Play, Play, Settle]
    assert events[-1] == Settle(60.0, GAME_ID, True)
    fake.calls.clear()
    assert feed.poll(120.0) == []
    assert fake.calls == ["scoreboard"]


def test_no_quotes_after_end_game_play(summary: dict[str, Any]) -> None:
    fake = Fake(summary)  # every play, through End Game, while ESPN still says "in"
    feed = fake.feed()
    assert not any(isinstance(e, Quote) for e in feed.poll(0.0))
    assert feed.poll(5.0) == []


def test_games_not_seen_in_progress_are_ignored(summary: dict[str, Any]) -> None:
    fake = Fake(summary)
    fake.games = [
        replace(fake.games[0], state="pre"),
        replace(fake.games[0], game_id="2", state="post", home_won=False),
    ]
    assert fake.feed().poll(0.0) == []
    assert fake.calls == ["scoreboard"]


def test_no_quote_until_the_play_backlog_has_gone_out(summary: dict[str, Any]) -> None:
    fake = Fake(None)  # the first play fetch fails
    feed = fake.feed()
    assert feed.poll(0.0) == []  # a quote alone would let the backlog trade next poll
    fake.summary = upto(summary, 5)
    assert [type(e) for e in feed.poll(5.0)] == [Play] * 5  # the backlog, still no quote
    assert [type(e) for e in feed.poll(10.0)] == [Quote]


def test_scoreboard_refreshed_at_most_every_minute(summary: dict[str, Any]) -> None:
    fake = Fake(upto(summary, 5))
    feed = fake.feed()
    for now in (0.0, 30.0, 59.0, 60.0, 119.0, 120.0):
        feed.poll(now)
    assert fake.calls.count("scoreboard") == 3
