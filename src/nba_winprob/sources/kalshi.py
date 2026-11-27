"""Kalshi one-minute candles for a game's market, cached, and parsed into ``Quote`` events."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nba_winprob.engine.events import Quote
from nba_winprob.sources.http import cached_json

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
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
