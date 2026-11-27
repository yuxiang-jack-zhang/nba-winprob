"""Kalshi one-minute candles for a game's market, cached, and parsed into ``Quote`` events;
the live top of book; and the ESPN game -> Kalshi ticker mapping.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from nba_winprob.engine.events import Quote
from nba_winprob.sources.http import cached_json, get_json

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
# ESPN -> Kalshi team codes, where they differ.
ESPN_TO_KALSHI = {"NY": "NYK", "SA": "SAS", "GS": "GSW", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
ET = ZoneInfo("America/New_York")
# Half an hour before tip to four hours after: every game ends inside it. At one
# candle a minute that is 270 candles, far under the endpoint's per-request cap.
BEFORE_TIP, AFTER_TIP = 1800, 4 * 3600


def candles(ticker: str, historical: bool, tip_ts: int, cache_dir: Path) -> dict[str, Any]:
    """Kalshi's raw candlestick response for the game window, cached per ticker."""
    # Settled markets move to the historical endpoint after a while; live ones are by series.
    base = f"/historical/markets/{ticker}" if historical else f"/series/KXNBAGAME/markets/{ticker}"
    url = (
        f"{KALSHI}{base}/candlesticks?start_ts={tip_ts - BEFORE_TIP}"
        f"&end_ts={tip_ts + AFTER_TIP}&period_interval=1"
    )
    return cached_json(cache_dir / f"{ticker}.json.gz", url, key="candlesticks")


def parse_candles(response: dict[str, Any], game_id: str) -> list[Quote]:
    """Each candle's closing book as a ``Quote``, skipping minutes with no two-sided book."""
    quotes = []
    for candle in response["candlesticks"]:
        # The historical endpoint says "close"/"volume", the live one "close_dollars"/"volume_fp".
        bid = candle["yes_bid"].get("close", candle["yes_bid"].get("close_dollars"))
        ask = candle["yes_ask"].get("close", candle["yes_ask"].get("close_dollars"))
        if bid is None or ask is None or not 0 < float(bid) <= float(ask) < 1:
            continue
        quotes.append(
            Quote(
                ts=float(candle["end_period_ts"]),
                game_id=game_id,
                ticker=response["ticker"],
                yes_bid=float(bid),
                yes_ask=float(ask),
                volume=float(candle.get("volume", candle.get("volume_fp", 0))),
            )
        )
    return quotes


def kalshi_ticker(tip_utc: str, away: str, home: str) -> str:
    """The home team's market for a game, from ESPN's tip time and team abbreviations.

    ``("2026-11-28T00:00Z", "MIA", "NY")`` -> ``"KXNBAGAME-26NOV27MIANYK-NYK"``: Kalshi
    dates games in New York time.
    """
    day = datetime.fromisoformat(tip_utc).astimezone(ET).strftime("%y%b%d").upper()
    away, home = ESPN_TO_KALSHI.get(away, away), ESPN_TO_KALSHI.get(home, home)
    return f"KXNBAGAME-{day}{away}{home}-{home}"


def parse_market(response: dict[str, Any], game_id: str, ts: float) -> Quote | None:
    """A ``GET /markets/{ticker}`` response's top of book, or None unless two-sided.

    An empty book reads bid 0, ask 1, as does a settled market.
    """
    market = response["market"]
    bid, ask = float(market["yes_bid_dollars"]), float(market["yes_ask_dollars"])
    if not 0 < bid <= ask < 1:
        return None
    return Quote(
        ts=ts, game_id=game_id, ticker=market["ticker"], yes_bid=bid, yes_ask=ask, volume=0
    )


def live_quote(ticker: str, game_id: str, ts: float) -> Quote | None:
    """The market's top of book now, stamped ``ts``. Public endpoint, no auth."""
    return parse_market(get_json(f"{KALSHI}/markets/{ticker}"), game_id, ts)
