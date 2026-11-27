"""ESPN and Kalshi parsing, the cache's retry rule, and a game's event stream.

Fixtures are trimmed from real game 401811001 (PHI at SAS, 2026-04-06): its first
quarter's scoring plays, timeouts and period end, the first few other plays, and
End Game. The header's score is set to the score at the end of that quarter, and End
Game's wallclock moved to a minute after it, so the trimmed game is continuous.
"""

from __future__ import annotations

import copy
import gzip
import itertools
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nba_winprob.config import Settings
from nba_winprob.engine.events import Play, Quote, Settle, sort_key
from nba_winprob.ingame.history import load_game
from nba_winprob.sources import http
from nba_winprob.sources.espn import parse_clock, parse_plays
from nba_winprob.sources.kalshi import parse_candles

FIXTURES = Path(__file__).parent / "fixtures"
GAME_ID = "401811001"
TICKER = "KXNBAGAME-26APR06PHISAS-SAS"


@pytest.fixture
def summary() -> dict[str, Any]:
    return json.loads((FIXTURES / f"espn_{GAME_ID}.json").read_text())


@pytest.fixture
def candles() -> dict[str, Any]:
    return json.loads((FIXTURES / f"kalshi_{TICKER}.json").read_text())


@pytest.mark.parametrize(
    ("display", "seconds"), [("12:00", 720.0), ("5:22", 322.0), ("38.2", 38.2), ("0.0", 0.0)]
)
def test_parse_clock(display: str, seconds: float) -> None:
    assert parse_clock(display) == seconds


def test_parse_plays(summary: dict[str, Any]) -> None:
    plays = parse_plays(summary, GAME_ID)
    assert len(plays) == len(summary["plays"])

    first = plays[0]  # the tip: wallclock 2026-04-07T00:11:09Z, SAS (id 24) at home
    assert (first.ts, first.seq, first.period, first.clock) == (1775520669.0, 4, 1, 720.0)
    assert first.team == "home"

    george_three = next(p for p in plays if p.seq == 17)
    assert george_three.kind == "score"
    assert (george_three.team, george_three.points) == ("away", 3)
    assert (george_three.home_score, george_three.away_score) == (0, 3)

    kinds = [p.kind for p in plays]
    assert kinds.count("score") == 29
    assert kinds.count("timeout") == 2
    assert kinds[-2:] == ["end_period", "end_game"]
    assert {p.team for p in plays if p.kind == "timeout"} == {"home", "away"}
    assert plays[-1].team is None
    assert all(p.points == 0 for p in plays if p.kind != "score")
    assert (plays[-1].home_score, plays[-1].away_score) == (29, 28)


def test_plays_follow_wallclock_and_ignore_stale_espn_scores(summary: dict[str, Any]) -> None:
    """ESPN's list order and its per-play scores can both be wrong; neither may matter."""
    expected = parse_plays(summary, GAME_ID)
    scrambled = copy.deepcopy(summary)
    scrambled["plays"].reverse()
    for play in scrambled["plays"]:
        play["homeScore"] = play["awayScore"] = 0
    assert parse_plays(scrambled, GAME_ID) == expected


def test_parse_plays_rejects_points_that_do_not_sum_to_the_final(
    summary: dict[str, Any],
) -> None:
    play = next(p for p in summary["plays"] if p["scoringPlay"])
    play["scoreValue"] += 1
    with pytest.raises(ValueError, match="ESPN final"):
        parse_plays(summary, GAME_ID)


@pytest.mark.parametrize(("first_shifted", "match"), [(0, "from tip"), (20, "wallclock jumps")])
def test_parse_plays_rejects_shifted_wallclocks(
    summary: dict[str, Any], first_shifted: int, match: str
) -> None:
    """Seen in real games: wallclocks a day late (Oct 2025) or an hour off (DST days)."""
    for play in itertools.islice(summary["plays"], first_shifted, None):
        when = datetime.fromisoformat(play["wallclock"]) + timedelta(hours=1)
        play["wallclock"] = when.isoformat()
    with pytest.raises(ValueError, match=match):
        parse_plays(summary, GAME_ID)


def test_parse_candles_keeps_only_two_sided_books(candles: dict[str, Any]) -> None:
    quotes = parse_candles(candles, GAME_ID)
    # Three of the nine candles are post-game 0.99/1.00 books: an ask of $1 is no book.
    assert len(quotes) == 6
    assert all(0 < q.yes_bid <= q.yes_ask < 1 for q in quotes)
    assert quotes[0] == Quote(1775520600.0, GAME_ID, TICKER, 0.76, 0.77, 505.0)


def test_parse_candles_reads_the_live_endpoint_shape() -> None:
    live = {
        "ticker": TICKER,
        "candlesticks": [
            {
                "end_period_ts": 60,
                "yes_bid": {"close_dollars": "0.6000"},
                "yes_ask": {"close_dollars": "0.6100"},
                "volume_fp": "2448.44",
            },
            {"end_period_ts": 120, "yes_bid": {"close": None}, "yes_ask": {"close": "0.5"}},
            {"end_period_ts": 180, "yes_bid": {"close": "0"}, "yes_ask": {"close": "0.5"}},
        ],
    }
    assert parse_candles(live, GAME_ID) == [Quote(60.0, GAME_ID, TICKER, 0.60, 0.61, 2448.44)]


def test_empty_cache_entry_is_refetched_and_empty_response_never_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "g.json.gz"
    path.write_bytes(gzip.compress(b'{"plays": []}'))
    monkeypatch.setattr(http, "get_json", lambda url: {"plays": [1]})
    assert http.cached_json(path, "url", "plays") == {"plays": [1]}
    assert json.loads(gzip.decompress(path.read_bytes())) == {"plays": [1]}

    empty = tmp_path / "empty.json.gz"
    monkeypatch.setattr(http, "get_json", lambda url: {"plays": []})
    with pytest.raises(ValueError, match="no plays"):
        http.cached_json(empty, "url", "plays")
    assert not empty.exists()


def test_load_game_is_one_sorted_stream(
    tmp_path: Path, summary: dict[str, Any], candles: dict[str, Any]
) -> None:
    after_buzzer = candles["candlesticks"][6]
    after_buzzer["yes_ask"]["close"] = "0.9950"  # a valid book, but the game is over
    assert len(parse_candles(candles, GAME_ID)) == 7
    for sub, name, data in (("espn", GAME_ID, summary), ("kalshi", TICKER, candles)):
        (tmp_path / "ingame" / sub).mkdir(parents=True)
        (tmp_path / "ingame" / sub / f"{name}.json.gz").write_bytes(
            gzip.compress(json.dumps(data).encode())
        )
    (tmp_path / "papertrade").mkdir()
    index = {"ticker": TICKER, "game_id": GAME_ID, "historical": True, "home_won": True}
    pd.DataFrame([{**index, "tip_ts": 1775520000}]).to_parquet(
        tmp_path / "papertrade" / "kalshi.parquet"
    )

    events = load_game(GAME_ID, Settings(data_dir=tmp_path))

    assert events == sorted(events, key=sort_key)
    end_game = next(e for e in events if isinstance(e, Play) and e.kind == "end_game")
    assert events[-1] == Settle(ts=end_game.ts, game_id=GAME_ID, home_won=True)
    quotes = [e for e in events if isinstance(e, Quote)]
    assert len(quotes) == 6  # the post-buzzer quote is gone
    assert all(q.ts <= end_game.ts for q in quotes)
    assert sum(isinstance(e, Play) for e in events) == len(summary["plays"])
