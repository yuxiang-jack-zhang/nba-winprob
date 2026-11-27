"""Run a strategy on live games: feed -> broker -> strategy, every step persisted.

The loop is the backtest's loop with a live feed: each event goes to the broker
first (it may fill pending orders), then to the strategy. Events are processed in
the order the feed returns them and never re-sorted: live events are all stamped
with receipt time, and the feed already orders each poll (see ``sources/live.py``).

Every event, order, fill and status change is written to Postgres, which is what
the catch-up agent reads. Stopping (a stop request, the feed running out, or a
crash) closes the run with its reason and cancels orders that never filled.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Protocol

import psycopg

from nba_winprob.db import live_store
from nba_winprob.engine.events import Broker, Event, Play, Settle, Strategy
from nba_winprob.features.ingame import game_seconds_elapsed
from nba_winprob.sources.espn import Game

log = logging.getLogger(__name__)


class Feed(Protocol):
    @property
    def games(self) -> Sequence[Game]: ...

    def poll(self, now: float) -> list[Event]: ...


def run_live(
    strategy: Strategy,
    broker: Broker,
    feed: Feed,
    conn: psycopg.Connection,
    run_id: int,
    *,
    poll_s: float = 5.0,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    should_stop: Callable[[], bool] = lambda: False,
    max_polls: int | None = None,
) -> None:
    """Poll until stopped. ``max_polls`` bounds the loop (tests, timed sessions)."""
    order_ids: dict[str, int] = {}  # Order.uid -> orders.id
    latest: dict[str, float] = {}  # game -> game seconds of the newest play written to game_state
    reason = "feed exhausted"
    try:
        polls = 0
        while max_polls is None or polls < max_polls:
            if should_stop():
                reason = "stop requested"
                break
            now = clock()
            events = feed.poll(now)
            live_store.record_events(conn, run_id, events, received_at=now)
            teams = {g.game_id: (g.home, g.away) for g in feed.games}
            for event in events:
                _step(conn, run_id, strategy, broker, event, order_ids)
                _update_state(conn, event, teams, latest)
            polls += 1
            sleep(poll_s)
    except BaseException as exc:  # KeyboardInterrupt included: record why the run ended
        reason = "stop requested" if isinstance(exc, KeyboardInterrupt) else f"error: {exc!r}"
        try:
            _close(conn, run_id, reason)
        except Exception:  # e.g. the connection died too: keep the original error
            log.exception("could not close run %d", run_id)
        raise
    _close(conn, run_id, reason)


def _step(
    conn: psycopg.Connection,
    run_id: int,
    strategy: Strategy,
    broker: Broker,
    event: Event,
    order_ids: dict[str, int],
) -> None:
    fills = broker.process(event)
    if isinstance(event, Settle):  # the broker dropped this game's unfilled orders
        conn.execute(
            "update orders set status = 'cancelled', error = 'game settled before it filled'"
            " where run_id = %s and game_id = %s and status = 'pending'",
            [run_id, event.game_id],
        )
    for fill in fills:
        live_store.record_fill(conn, order_ids[fill.order.uid], fill)
        log.info(
            "fill %s %+d %s @ %.2f",
            fill.order.game_id,
            fill.order.contracts,
            fill.order.team,
            fill.price,
        )
        strategy.on_fill(fill)
    for order in strategy.on_event(event, broker.positions):
        order_id = live_store.record_order(conn, run_id, order)
        order_ids[order.uid] = order_id
        log.info("order %s %+d %s: %s", order.game_id, order.contracts, order.team, order.reason)
        try:
            broker.submit(order)
        except ValueError as err:  # a refused order is a fact to record, not a reason to stop
            live_store.set_order_status(conn, order_id, "rejected", str(err))


def _update_state(
    conn: psycopg.Connection,
    event: Event,
    teams: dict[str, tuple[str, str]],
    latest: dict[str, float],
) -> None:
    if event.game_id not in teams:
        return
    if isinstance(event, Play):
        # A late-inserted play carries an older score; only move game_state forward.
        elapsed = game_seconds_elapsed(event.period, event.clock)
        if elapsed < latest.get(event.game_id, float("-inf")):
            return
        latest[event.game_id] = elapsed
    live_store.update_game_state(conn, event, *teams[event.game_id])


def _close(conn: psycopg.Connection, run_id: int, reason: str) -> None:
    conn.execute(
        "update orders set status = 'cancelled', error = 'not filled when the run stopped'"
        " where run_id = %s and status = 'pending'",
        [run_id],
    )
    live_store.stop_run(conn, run_id, reason)
    log.info("run %d stopped: %s", run_id, reason)
