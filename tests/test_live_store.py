"""Live trading storage (migration 0002) on a real, fresh Postgres per test (D7)."""

from __future__ import annotations

from dataclasses import asdict

import psycopg
import pytest
from psycopg import errors

from nba_winprob.db import live_store, migrate
from nba_winprob.engine.events import Fill, Order, Play, Quote, Settle

G = "401585001"


def _play(ts: float, home: int, away: int, kind: str = "score", text: str = "") -> Play:
    return Play(ts, G, int(ts), 2, 300.0, home, away, "home", kind, 2, text)  # type: ignore[arg-type]


def _order(ts: float, contracts: int, team: str = "home", game: str = G) -> Order:
    return Order(ts, game, team, contracts, f"why {ts}", {"run": 12, "timeout": True})  # type: ignore[arg-type]


def _filled(db: psycopg.Connection, run: int, order: Order, price: float = 0.4) -> int:
    oid = live_store.record_order(db, run, order)
    live_store.record_fill(db, oid, Fill(order, order.ts + 30, price, 0.02))
    return oid


@pytest.fixture
def run(db: psycopg.Connection) -> int:
    return live_store.start_run(db, "run_reversal", "paper", {"latency": 30})


def test_migration_0002_applies_after_0001(empty_db: psycopg.Connection) -> None:
    assert migrate.apply_migrations(empty_db)[:2] == ["0001_base_tables", "0002_live"]
    recorded = [r[0] for r in empty_db.execute("select version from schema_migrations")]
    assert "0002_live" in recorded


@pytest.mark.parametrize(
    "statement",
    [
        "insert into runs (strategy, broker) values ('s', 'real')",
        "insert into events (run_id, ts, game_id, kind, payload) values (%s, 0, 'g', 'tick', '{}')",
        "insert into orders (run_id, ts, game_id, team, contracts, reason)"
        " values (%s, 0, 'g', 'both', 1, 'r')",
        "insert into orders (run_id, ts, game_id, team, contracts, reason)"
        " values (%s, 0, 'g', 'home', 0, 'r')",
    ],
)
def test_constraints_reject_bad_rows(db: psycopg.Connection, run: int, statement: str) -> None:
    params = [run] if "%s" in statement else []
    with pytest.raises(errors.CheckViolation):
        db.execute(statement, params)


def test_fill_price_above_one_is_rejected(db: psycopg.Connection, run: int) -> None:
    order = _order(1.0, 5)
    oid = live_store.record_order(db, run, order)
    with pytest.raises(errors.CheckViolation):
        live_store.record_fill(db, oid, Fill(order, 2.0, 1.01, 0.0))
    # The failed fill rolled back whole: the order was not marked filled.
    assert live_store.recent_orders(db, run, 1)[0]["status"] == "pending"


def test_stop_run_once(db: psycopg.Connection, run: int) -> None:
    live_store.stop_run(db, run, "ctrl-c")
    assert db.execute("select stop_reason from runs where id = %s", [run]).fetchone() == ("ctrl-c",)
    with pytest.raises(ValueError, match="not a running run"):
        live_store.stop_run(db, run, "again")


def test_events_round_trip(db: psycopg.Connection, run: int) -> None:
    events = [
        _play(100.0, 2, 0, text="Jokic makes 2-foot dunk"),
        Quote(101.5, G, "KXNBAGAME-X", 0.41, 0.43, 0.0),
        Settle(9000.0, G, True),
    ]
    live_store.record_events(db, run, events, received_at=123.25)
    rows = db.execute(
        "select ts, game_id, kind, payload from events where run_id = %s order by seq", [run]
    ).fetchall()
    assert [r[2] for r in rows] == ["play", "quote", "settle"]
    assert {r[0] for r in rows} == {123.25} and {r[1] for r in rows} == {G}
    rebuilt = [type(e)(**r[3]) for e, r in zip(events, rows, strict=True)]
    assert rebuilt == events
    assert [asdict(e) for e in rebuilt] == [r[3] for r in rows]


def test_fill_marks_order_filled(db: psycopg.Connection, run: int) -> None:
    other = live_store.record_order(db, run, _order(1.0, 3))
    oid = _filled(db, run, _order(2.0, 7), price=0.55)
    statuses = dict(db.execute("select id, status from orders").fetchall())
    assert statuses == {other: "pending", oid: "filled"}
    fill = db.execute("select contracts, price from fills where order_id = %s", [oid])
    assert fill.fetchone() == (7, 0.55)


def test_set_order_status(db: psycopg.Connection, run: int) -> None:
    oid = live_store.record_order(db, run, _order(1.0, 3))
    live_store.set_order_status(db, oid, "rejected", "insufficient funds")
    assert db.execute("select status, error from orders").fetchone() == (
        "rejected",
        "insufficient funds",
    )
    with pytest.raises(errors.CheckViolation):
        live_store.set_order_status(db, oid, "done")
    with pytest.raises(ValueError, match="no order"):
        live_store.set_order_status(db, oid + 1, "cancelled")


def test_open_positions_net_per_game_and_team(db: psycopg.Connection, run: int) -> None:
    _filled(db, run, _order(1.0, 10))
    _filled(db, run, _order(2.0, -4))  # partly sold: 6 left
    _filled(db, run, _order(3.0, 5, team="away"))
    _filled(db, run, _order(4.0, 8, game="g2"))
    _filled(db, run, _order(5.0, -8, game="g2"))  # flat: dropped
    live_store.record_order(db, run, _order(6.0, 99))  # unfilled: not a position
    other_run = live_store.start_run(db, "other", "paper", {})
    _filled(db, other_run, _order(7.0, 50))
    assert live_store.open_positions(db, run) == {(G, "home"): 6, (G, "away"): 5}


def test_game_state_upserts(db: psycopg.Connection) -> None:
    live_store.update_game_state(db, Quote(1.0, G, "T", 0.40, 0.42, 0.0), "DEN", "LAL")
    (first,) = live_store.game_states(db)
    assert first["status"] == "scheduled" and first["home_score"] is None
    assert (first["yes_bid"], first["yes_ask"]) == (0.40, 0.42)

    live_store.update_game_state(db, _play(2.0, 10, 8, text="Murray 3pt"), "DEN", "LAL")
    live_store.update_game_state(db, _play(3.0, 12, 8, text="Jokic layup"), "DEN", "LAL")
    (state,) = live_store.game_states(db)
    assert (state["home_score"], state["away_score"], state["last_play"]) == (12, 8, "Jokic layup")
    assert (state["home"], state["away"], state["status"]) == ("DEN", "LAL", "live")
    assert (state["yes_bid"], state["yes_ask"]) == (0.40, 0.42)  # kept from the quote
    assert state["updated_at"] >= first["updated_at"]

    live_store.update_game_state(db, Settle(4.0, G, True), "DEN", "LAL")
    (final,) = live_store.game_states(db)
    assert final["status"] == "final" and final["home_score"] == 12

    live_store.update_game_state(db, _play(5.0, 14, 8, text="late entry"), "DEN", "LAL")
    (late,) = live_store.game_states(db)
    assert late["status"] == "final" and late["home_score"] == 14


def test_recent_orders_newest_first_with_fill(db: psycopg.Connection, run: int) -> None:
    filled = _filled(db, run, _order(1.0, 10), price=0.37)
    pending = live_store.record_order(db, run, _order(5.0, -10))
    cancelled = live_store.record_order(db, run, _order(3.0, 2))
    live_store.set_order_status(db, cancelled, "cancelled")

    rows = live_store.recent_orders(db, run, 2)
    assert [r["id"] for r in rows] == [pending, cancelled]
    assert rows[0]["fill_price"] is None and rows[0]["status"] == "pending"

    (oldest,) = live_store.recent_orders(db, run, 3)[2:]
    assert oldest["id"] == filled
    assert oldest["status"] == "filled" and oldest["fill_price"] == pytest.approx(0.37)
    assert oldest["fee"] == pytest.approx(0.02)
    assert oldest["reason"] == "why 1.0" and oldest["meta"] == {"run": 12, "timeout": True}
    assert (oldest["game_id"], oldest["team"], oldest["contracts"]) == (G, "home", 10)
