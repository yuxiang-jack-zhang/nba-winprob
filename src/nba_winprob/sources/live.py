"""Live events for paper trading: ESPN plays, Kalshi quotes and settlement, by polling.

The runner calls ``LiveFeed.poll(time.time())`` every few seconds and passes the
returned events, in the order given, to the broker and then the strategy.

**Every event is stamped ``ts = now``, the receipt time, not ESPN's wallclock.** Live,
a play's wallclock is earlier than quotes already processed (ESPN posts plays late),
and the broker rejects orders stamped before the current event. Receipt time is when
the strategy could really have known, so it is the honest decision time. Within one
poll a game's new plays keep replay order (wallclock, then sequence number) and come
before that poll's quote, then its ``Settle``. Do not re-sort a poll's events with
``sort_key``: at equal ``ts`` it orders plays by sequence number, which puts a
late-entered play after plays that happened after it.

Scores are rebuilt as in history (D10): every poll parses the game's full play list in
wallclock order, so a play ESPN inserts late carries the right running score, and so
does every play emitted after it. Limitation: plays emitted earlier are not re-sent, so
their scores stay as they were when emitted.

A failed fetch is logged and retried next poll; one game's bad data never stops another.

**No quote until a game's plays have been caught up.** The first successful play fetch
replays the whole backlog at once, all stamped ``now``; a run long over in game time can
look like a fresh trigger while it streams through. Quotes for a game start the poll after
that catch-up, so no strategy has a price to act on backlog plays. A game whose first
fetch fails stays quote-less until a fetch succeeds.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from datetime import date, datetime, timedelta
from typing import Any

from nba_winprob.engine.events import Event, Play, Quote, Settle
from nba_winprob.sources import espn, kalshi

log = logging.getLogger(__name__)

SCOREBOARD_EVERY = 60.0  # seconds
# The scoreboard day turns over at 10 am New York time: the last games end around 1:30 am
# and the earliest tip around noon.
DAY_TURNOVER = timedelta(hours=10)


class LiveFeed:
    """Polls ESPN and Kalshi; the fetch functions are injectable so tests need no network."""

    def __init__(
        self,
        scoreboard: Callable[[date], list[espn.Game]] = espn.scoreboard,
        summary: Callable[[str], dict[str, Any]] = espn.live_summary,
        quote: Callable[[str, str, float], Quote | None] = kalshi.live_quote,
    ) -> None:
        self._scoreboard, self._summary, self._quote = scoreboard, summary, quote
        self._games: list[espn.Game] = []
        self._board_at = float("-inf")
        self._emitted: dict[str, set[int]] = {}  # games seen in progress -> play seqs emitted
        self._caught_up: set[str] = set()  # games whose play backlog went out in an earlier poll
        self._ended: set[str] = set()  # End Game emitted: later quotes are not tradeable
        self._done: set[str] = set()

    @property
    def games(self) -> list[espn.Game]:
        """Today's games as of the last scoreboard refresh."""
        return self._games

    def poll(self, now: float) -> list[Event]:
        if now - self._board_at >= SCOREBOARD_EVERY:
            day = (datetime.fromtimestamp(now, kalshi.ET) - DAY_TURNOVER).date()
            try:
                self._games = self._scoreboard(day)
                self._board_at = now
            except Exception:
                log.exception("scoreboard %s failed", day)
        events: list[Event] = []
        for game in self._games:
            gid = game.game_id
            if gid in self._done or (game.state != "in" and gid not in self._emitted):
                continue
            caught_up = gid in self._caught_up
            plays = self._new_plays(gid, now)
            events += plays
            if game.state == "in" and gid not in self._ended and caught_up:
                ticker = kalshi.kalshi_ticker(game.tip_utc, game.away, game.home)
                try:
                    quote = self._quote(ticker, gid, now)
                except Exception:
                    log.exception("quote %s failed", ticker)
                    quote = None
                if quote is not None:
                    events.append(quote)
            if game.state == "post":
                self._done.add(gid)
                if game.home_won is not None:  # a postponed game never settles here
                    events.append(Settle(ts=now, game_id=gid, home_won=game.home_won))
        return events

    def _new_plays(self, gid: str, now: float) -> list[Play]:
        emitted = self._emitted.setdefault(gid, set())
        try:
            summary = self._summary(gid)
            plays = (
                espn.parse_plays(summary, gid, check_final=False) if summary.get("plays") else []
            )
        except Exception:
            log.exception("summary %s failed", gid)
            return []
        self._caught_up.add(gid)  # quotes may follow from the next poll on
        new = [replace(p, ts=now) for p in plays if p.seq not in emitted]
        emitted.update(p.seq for p in new)
        if any(p.kind == "end_game" for p in new):
            self._ended.add(gid)
        return new
