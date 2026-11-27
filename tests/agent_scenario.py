"""One evening of paper trading, written through the real live_store writes.

Shared by the agent's tool tests (which pin every number) and its opt-in evals (which
check the agent reports those numbers), so the evals' ground truth is itself tested.

- MIA @ NY (live, Q3 4:12, MIA 72-70): MIA ran 12-2, NY took a timeout, and the bot
  bought 200 NY at 0.40. The book is now 0.44 / 0.46.
- DEN @ BOS (final, BOS 110-104): the bot bought 150 DEN at 0.30 and lost.
- A second NY buy was rejected by the broker.
"""

from __future__ import annotations

import psycopg

from nba_winprob.db import live_store
from nba_winprob.engine.events import Fill, Order, Play, Quote, Settle, kalshi_taker_fee

T0 = 1_795_000_000.0  # run start; NOW is an hour later
NOW = T0 + 3600
NYK, BOS = "401810001", "401810002"
NY_REASON = "MIA ran 12-2 in the last 4:00 with the margin at 2; buying NY, the team run on"
DEN_REASON = "BOS ran 11-0 in the last 4:00 with the margin at 3; buying DEN, the team run on"

FEE_NY = kalshi_taker_fee(0.40, 200)
FEE_DEN = kalshi_taker_fee(0.30, 150)
# Cash out: 200 x 0.40 + 150 x 0.30 + fees. Worth now: 200 NY at the 0.44 bid; DEN lost.
PNL = round(-(80.0 + FEE_NY) - (45.0 + FEE_DEN) + 200 * 0.44, 2)


def _play(ts: float, game: str, home: int, away: int, clock: float, team: str, text: str) -> Play:
    return Play(ts, game, int(ts), 3 if game == NYK else 4, clock, home, away, team, "x", 0, text)


def seed(conn: psycopg.Connection) -> int:
    run_id = live_store.start_run(conn, "run_reversal_v1", "paper", {"min_run_net": 10})
    teams = {NYK: ("NY", "MIA"), BOS: ("BOS", "DEN")}
    events = [
        _play(T0 + 600, BOS, 90, 93, 400, "home", "BOS 3-pointer"),
        Quote(T0 + 610, BOS, "KXNBAGAME-BOS", 0.68, 0.70, 0),
        _play(T0 + 1500, BOS, 110, 104, 0, "home", "End of game"),
        Settle(T0 + 1510, BOS, home_won=True),
        _play(T0 + 2900, NYK, 68, 62, 400, "home", "NY layup"),
        _play(T0 + 3000, NYK, 70, 72, 270, "away", "MIA 3-pointer"),
        _play(T0 + 3010, NYK, 70, 72, 252, "home", "NY full timeout"),
        Quote(T0 + 3020, NYK, "KXNBAGAME-NYK", 0.39, 0.40, 0),
        Quote(T0 + 3500, NYK, "KXNBAGAME-NYK", 0.44, 0.46, 0),
    ]
    for e in events:
        live_store.record_events(conn, run_id, [e], received_at=e.ts)
        live_store.update_game_state(conn, e, *teams[e.game_id])

    def trade(order: Order, price: float | None, error: str | None = None) -> None:
        order_id = live_store.record_order(conn, run_id, order)
        if price is not None:
            fee = kalshi_taker_fee(price, order.contracts)
            live_store.record_fill(conn, order_id, Fill(order, order.ts + 5, price, fee))
        else:
            live_store.set_order_status(conn, order_id, "rejected", error)

    meta_den = {"run_net": 11, "margin": 3, "timeout": False}
    trade(Order(T0 + 620, BOS, "away", 150, DEN_REASON, meta_den), 0.30)
    meta_ny = {"run_net": 10, "margin": 2, "timeout": True}
    trade(Order(T0 + 3015, NYK, "home", 200, NY_REASON, meta_ny), 0.40)
    trade(
        Order(T0 + 3016, NYK, "home", 50, "duplicate entry"),
        None,
        "order at ts 1795003016 is before the last event",
    )
    return run_id
