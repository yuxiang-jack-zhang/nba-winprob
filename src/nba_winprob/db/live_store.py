"""Persist live trading runs (migration 0002): events, orders, fills and game state.

Each write is its own transaction. Connections are autocommit (``store.connect``).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from nba_winprob.engine.events import Event, Fill, Order, Play, Quote, Settle, Side

_KIND = {Play: "play", Quote: "quote", Settle: "settle"}
# game_state columns an event may leave null, keeping what an earlier event set.
_STATE_FIELDS = ("home_score", "away_score", "period", "clock", "last_play", "yes_bid", "yes_ask")


def start_run(
    conn: psycopg.Connection, strategy: str, broker: str, params: Mapping[str, Any]
) -> int:
    with conn.transaction():
        row = conn.execute(
            "insert into runs (strategy, broker, params) values (%s, %s, %s) returning id",
            [strategy, broker, Jsonb(dict(params))],
        ).fetchone()
    assert row is not None
    return int(row[0])


def stop_run(conn: psycopg.Connection, run_id: int, reason: str) -> None:
    """Raises ``ValueError`` if the run does not exist or was already stopped."""
    with conn.transaction():
        cur = conn.execute(
            "update runs set stopped_at = now(), stop_reason = %s"
            " where id = %s and stopped_at is null",
            [reason, run_id],
        )
        if cur.rowcount != 1:
            raise ValueError(f"run {run_id} is not a running run")


def record_events(
    conn: psycopg.Connection,
    run_id: int,
    events: Iterable[Event],
    received_at: float | None = None,
) -> None:
    """Append ``events`` in order, stamped with when they were received (default: now)."""
    received_at = time.time() if received_at is None else received_at
    with (
        conn.transaction(),
        conn.cursor().copy("copy events (run_id, ts, game_id, kind, payload) from stdin") as copy,
    ):
        for e in events:
            copy.write_row([run_id, received_at, e.game_id, _KIND[type(e)], Jsonb(asdict(e))])


def record_order(conn: psycopg.Connection, run_id: int, order: Order) -> int:
    with conn.transaction():
        row = conn.execute(
            "insert into orders (run_id, ts, game_id, team, contracts, reason, meta)"
            " values (%s, %s, %s, %s, %s, %s, %s) returning id",
            [
                run_id,
                order.ts,
                order.game_id,
                order.team,
                order.contracts,
                order.reason,
                Jsonb(dict(order.meta)),
            ],
        ).fetchone()
    assert row is not None
    return int(row[0])


def set_order_status(
    conn: psycopg.Connection, order_id: int, status: str, error: str | None = None
) -> None:
    with conn.transaction():
        cur = conn.execute(
            "update orders set status = %s, error = %s where id = %s", [status, error, order_id]
        )
        if cur.rowcount != 1:
            raise ValueError(f"no order {order_id}")


def record_fill(conn: psycopg.Connection, order_id: int, fill: Fill) -> None:
    """Store the fill and mark its order filled, together."""
    with conn.transaction():
        conn.execute(
            "insert into fills (order_id, ts, price, fee, contracts) values (%s, %s, %s, %s, %s)",
            [order_id, fill.ts, fill.price, fill.fee, fill.order.contracts],
        )
        conn.execute("update orders set status = 'filled' where id = %s", [order_id])


def update_game_state(conn: psycopg.Connection, event: Event, home: str, away: str) -> None:
    """Upsert the game's row from one event; fields the event does not carry are kept.

    A play sets score, period, clock and last play; a quote sets the book; either
    ``end_game`` or a settlement marks the game final.
    """
    p: dict[str, Any] = dict.fromkeys((*_STATE_FIELDS, "status"))
    p.update(game_id=event.game_id, home=home, away=away)
    if isinstance(event, Play):
        p.update(
            home_score=event.home_score,
            away_score=event.away_score,
            period=event.period,
            clock=event.clock,
            status="final" if event.kind == "end_game" else "live",
            last_play=event.text,
        )
    elif isinstance(event, Quote):
        p.update(yes_bid=event.yes_bid, yes_ask=event.yes_ask)
    else:
        p["status"] = "final"
    cols = ", ".join(p)
    values = ", ".join(f"%({c})s" for c in p).replace(
        "%(status)s", "coalesce(%(status)s, 'scheduled')"
    )
    kept = ", ".join(f"{c} = coalesce(%({c})s, game_state.{c})" for c in _STATE_FIELDS)
    with conn.transaction():
        conn.execute(
            f"insert into game_state ({cols}) values ({values}) on conflict (game_id) do update"
            f" set home = excluded.home, away = excluded.away, updated_at = now(), {kept},"
            # A late-entered play must not reopen a finished game.
            " status = case when game_state.status = 'final' then 'final'"
            " else coalesce(%(status)s, game_state.status) end",
            p,
        )


# --------------------------------------------------------------------------- reads


def open_positions(conn: psycopg.Connection, run_id: int) -> dict[tuple[str, Side], int]:
    """Net filled contracts per (game, team); flat positions are left out."""
    rows = conn.execute(
        "select o.game_id, o.team, sum(f.contracts) from fills f"
        " join orders o on o.id = f.order_id where o.run_id = %s"
        " group by o.game_id, o.team having sum(f.contracts) <> 0",
        [run_id],
    ).fetchall()
    return {(game_id, team): int(n) for game_id, team, n in rows}


def recent_orders(conn: psycopg.Connection, run_id: int, limit: int) -> list[dict[str, Any]]:
    """Newest first, with the size-weighted fill price and total fee (null if unfilled)."""
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            "select o.id, o.ts, o.game_id, o.team, o.contracts, o.reason, o.meta,"
            " o.status, o.error, f.fill_price, f.fee from orders o"
            " left join lateral (select sum(price * abs(contracts)) / sum(abs(contracts))"
            " as fill_price, sum(fee) as fee from fills where order_id = o.id) f on true"
            " where o.run_id = %s order by o.ts desc, o.id desc limit %s",
            [run_id, limit],
        ).fetchall()


def game_states(conn: psycopg.Connection) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute("select * from game_state order by game_id").fetchall()
