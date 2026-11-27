"""The games to replay, and each one's events from the cached ESPN and Kalshi data."""

from __future__ import annotations

import pandas as pd

from nba_winprob.config import Settings
from nba_winprob.engine.events import Event, Settle, sort_key
from nba_winprob.sources import espn, kalshi


def game_index(settings: Settings) -> pd.DataFrame:
    """One row per Kalshi-matched game (``papertrade/fetch.py kalshi`` builds the file)."""
    return pd.read_parquet(settings.data_dir / "papertrade" / "kalshi.parquet")


def load_game(game_id: str, settings: Settings) -> list[Event]:
    """A game's plays, quotes and settlement in replay order; fetches what is not cached."""
    games = game_index(settings)
    game = games[games["game_id"] == game_id].iloc[0]
    plays = espn.parse_plays(espn.summary(game_id, settings.ingame_dir / "espn"), game_id)
    response = kalshi.candles(
        game.ticker, game.historical, game.tip_ts, settings.ingame_dir / "kalshi"
    )
    end = next((p.ts for p in plays if p.kind == "end_game"), plays[-1].ts)
    # After the final buzzer the outcome is known: later quotes are not tradeable signal.
    quotes = [q for q in kalshi.parse_candles(response, game_id) if q.ts <= end]
    settle = Settle(ts=end, game_id=game_id, home_won=bool(game.home_won))
    return sorted([*plays, *quotes, settle], key=sort_key)
