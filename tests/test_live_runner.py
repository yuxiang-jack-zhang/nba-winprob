"""The live runner: feed -> broker -> strategy, with every step persisted.

A scripted feed stands in for ESPN and Kalshi, so these run against real Postgres
but no network.
"""

from __future__ import annotations

import psycopg
import pytest

from nba_winprob.db import live_store
from nba_winprob.engine.broker import SimBroker
from nba_winprob.engine.events import Event, Fill, Order, Play, Positions, Quote, Settle
from nba_winprob.live.runner import run_live
from nba_winprob.sources.espn import Game

G = "401"
GAME = Game(
    game_id=G, tip_utc="2026-11-27T00:30Z", home="NY", away="MIA", state="in", home_won=None
)


def play(ts: float, home: int, away: int, clock: float = 600.0, text: str = "") -> Play:
    return Play(ts, G, int(ts), 1, clock, home, away, "home", "other", 0, text or f"t{ts}")


def quote(ts: float, bid: float = 0.40, ask: float = 0.42) -> Quote:
    return Quote(ts, G, "KXNBAGAME-26NOV26MIANYK-NYK", bid, ask, 0.0)


class Feed:
    """Returns one scripted batch per poll, like LiveFeed (events stamped with poll time)."""

    def __init__(self, batches: list[list[Event]]) -> None:
        self.batches = batches
        self.games = [GAME]

    def poll(self, now: float) -> list[Event]:
        return self.batches.pop(0) if self.batches else []


class Buyer:
    """Buys 10 home on the first play it sees, sells on the first quote after filling."""

    name = "buyer"

    def __init__(self, contracts: int = 10) -> None:
        self.contracts, self.bought, self.filled = contracts, False, False

    def on_event(self, event: Event, positions: Positions) -> list[Order]:
        if isinstance(event, Play) and not self.bought:
            self.bought = True
            return [Order(event.ts, G, "home", self.contracts, "test entry", {"why": "test"})]
        if isinstance(event, Quote) and self.filled and positions.get((G, "home")):
            self.filled = False
            return [Order(event.ts, G, "home", -positions[(G, "home")], "test exit")]
        return []

    def on_fill(self, fill: Fill) -> None:
        self.filled = fill.order.contracts > 0


def go(
    db: psycopg.Connection,
    batches: list[list[Event]],
    strategy: Buyer | None = None,
    stop_after: int | None = None,
) -> int:
    run_id = live_store.start_run(db, "buyer", "paper", {})
    polls = iter(range(100, 100 + 10 * (len(batches) + 1), 10))
    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return stop_after is not None and calls["n"] > stop_after

    run_live(
        strategy or Buyer(),
        SimBroker(latency_s=0),
        Feed(batches),
        db,
        run_id,
        clock=lambda: float(next(polls)),
        sleep=lambda _: None,
        should_stop=should_stop,
        max_polls=len(batches),
    )
    return run_id


def rows(db: psycopg.Connection, sql: str, *params: object) -> list[tuple[object, ...]]:
    return db.execute(sql, params).fetchall()


def test_events_orders_and_fills_are_persisted(db: psycopg.Connection) -> None:
    run_id = go(
        db,
        [
            [play(100, 2, 0), quote(100)],  # buy decided at 100
            [quote(110)],  # fills at the ask .42; the strategy then sells
            [quote(120, 0.45, 0.47)],  # sell fills at the bid .45
            [Settle(130, G, True)],
        ],
    )
    assert rows(
        db, "select kind, count(*) from events where run_id=%s group by 1 order by 1", run_id
    ) == [
        ("play", 1),
        ("quote", 3),
        ("settle", 1),
    ]
    orders = live_store.recent_orders(db, run_id, 10)
    assert [(o["contracts"], o["status"], o["fill_price"]) for o in orders] == [
        (-10, "filled", pytest.approx(0.45)),
        (10, "filled", pytest.approx(0.42)),
    ]
    assert orders[1]["reason"] == "test entry" and orders[1]["meta"] == {"why": "test"}
    assert live_store.open_positions(db, run_id) == {}
    stopped = rows(db, "select stop_reason from runs where id=%s", run_id)
    assert stopped == [("feed exhausted",)]


def test_game_state_ignores_a_late_play_with_an_older_score(db: psycopg.Connection) -> None:
    go(
        db,
        [
            [play(100, 10, 8, clock=300, text="newest")],
            [play(110, 9, 8, clock=320, text="late insert")],  # earlier in the game
        ],
    )
    [state] = live_store.game_states(db)
    assert (state["home"], state["away"]) == ("NY", "MIA")
    assert (state["home_score"], state["last_play"]) == (10, "newest")


def test_unfilled_orders_are_cancelled_when_the_run_stops(db: psycopg.Connection) -> None:
    run_id = go(db, [[play(100, 2, 0)]])  # no quote ever arrives
    [order] = live_store.recent_orders(db, run_id, 10)
    assert order["status"] == "cancelled"


def test_broker_rejection_is_recorded_not_raised(db: psycopg.Connection) -> None:
    class Seller(Buyer):
        def on_event(self, event: Event, positions: Positions) -> list[Order]:
            if isinstance(event, Play):
                return [Order(event.ts - 50, G, "home", 5, "stale")]  # backdated: broker refuses
            return []

    run_id = go(db, [[play(100, 2, 0)]], strategy=Seller())
    [order] = live_store.recent_orders(db, run_id, 10)
    assert order["status"] == "rejected" and "before" in order["error"]


def test_stop_request_stops_new_orders_and_records_why(db: psycopg.Connection) -> None:
    run_id = go(db, [[quote(100)], [play(110, 2, 0)]], stop_after=1)
    assert live_store.recent_orders(db, run_id, 10) == []
    assert rows(db, "select stop_reason from runs where id=%s", run_id) == [("stop requested",)]


def test_a_crash_still_closes_the_run(db: psycopg.Connection) -> None:
    class Broken(Buyer):
        def on_event(self, event: Event, positions: Positions) -> list[Order]:
            raise RuntimeError("strategy bug")

    with pytest.raises(RuntimeError):
        go(db, [[play(100, 2, 0)]], strategy=Broken())
    [(reason,)] = rows(db, "select stop_reason from runs")
    assert isinstance(reason, str) and reason.startswith("error: RuntimeError")


def test_demo_broker_path_end_to_end(db: psycopg.Connection) -> None:
    """run_live with the real DemoBroker against a fake exchange: a rejection, a dropped
    network on the fills poll, then a fill that lands on its order's row."""
    from nba_winprob.brokers.demo import DemoBroker
    from tests.test_kalshi_demo import FakeExchange, client, raw_fill

    exchange = FakeExchange()
    exchange.fills = [raw_fill("oid0", "10.00", "0.4200")]  # oid0: the first order placed

    class BuyOnCue(Buyer):
        def on_event(self, event: Event, positions: Positions) -> list[Order]:
            if isinstance(event, Play) and event.text == "buy":
                return [Order(event.ts, G, "home", 10, "cue")]
            return []

    def network(down: bool) -> None:
        exchange.get_error = "GET fills: URLError timed out" if down else None

    script: list[list[Event]] = [
        [play(100, 2, 0, text="buy"), quote(100)],  # no quote yet when the order goes in
        [play(110, 2, 0, text="buy")],  # placed: oid0
        [quote(120)],  # fills poll fails (set below): retried, the run goes on
        [quote(130)],  # fills poll succeeds
    ]

    class Scripted(Feed):
        def poll(self, now: float) -> list[Event]:
            if len(self.batches) == 2:
                network(down=True)
            if len(self.batches) == 1:
                network(down=False)
            return super().poll(now)

    run_id = live_store.start_run(db, "cue", "demo", {})
    polls = iter(range(100, 200, 10))
    run_live(
        BuyOnCue(),
        DemoBroker(client(exchange), poll_every_s=0, clock=lambda: 0.0),
        Scripted(script),
        db,
        run_id,
        clock=lambda: float(next(polls)),
        sleep=lambda _: None,
        max_polls=len(script),
    )
    filled, rejected = live_store.recent_orders(db, run_id, 10)
    assert (rejected["status"], rejected["error"]) == ("rejected", "no quote yet")
    assert (filled["status"], filled["fill_price"]) == ("filled", pytest.approx(0.42))
    assert live_store.open_positions(db, run_id) == {(G, "home"): 10}
    assert rows(db, "select stop_reason from runs where id=%s", run_id) == [("feed exhausted",)]


def test_settlement_cancels_that_games_unfilled_orders(db: psycopg.Connection) -> None:
    run_id = go(db, [[play(100, 2, 0)], [Settle(110, G, True)], [quote(120)]])
    [order] = live_store.recent_orders(db, run_id, 10)
    assert (order["status"], order["error"]) == ("cancelled", "game settled before it filled")


def test_a_failed_close_does_not_hide_the_original_error(
    db: psycopg.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken(Buyer):
        def on_event(self, event: Event, positions: Positions) -> list[Order]:
            raise RuntimeError("strategy bug")

    def dead(*args: object) -> None:
        raise psycopg.OperationalError("connection lost")

    monkeypatch.setattr(live_store, "stop_run", dead)
    with pytest.raises(RuntimeError, match="strategy bug"):
        go(db, [[play(100, 2, 0)]], strategy=Broken())
