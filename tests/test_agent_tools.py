"""The catch-up agent's read-only tools, against a seeded evening of paper trading."""

from __future__ import annotations

import json

import psycopg
import pytest

from nba_winprob.agent import tools
from nba_winprob.db import live_store
from nba_winprob.engine.events import Fill, Order, kalshi_taker_fee
from tests import agent_scenario as sc


@pytest.fixture
def run(db: psycopg.Connection) -> int:
    return sc.seed(db)


def test_run_status_marks_positions_and_totals_pnl(db: psycopg.Connection, run: int) -> None:
    status = tools.run_status(db, run)
    assert (status["strategy"], status["broker"], status["stopped"]) == (
        "run_reversal_v1",
        "paper",
        None,
    )
    by_team = {p["team"]: p for p in status["positions"]}
    assert by_team["NY"] == {
        "game_id": sc.NYK,
        "matchup": "MIA @ NY",
        "team": "NY",
        "contracts": 200,
        "net_cost": pytest.approx(80.0 + sc.FEE_NY),
        "mark": 0.44,  # home is marked at the YES bid
        "value": 88.0,
        "state": "open",
    }
    assert (by_team["DEN"]["mark"], by_team["DEN"]["state"]) == (0.0, "lost")
    assert status["fees_paid"] == pytest.approx(sc.FEE_NY + sc.FEE_DEN)
    assert status["pnl"] == pytest.approx(sc.PNL)
    by_game = {g["matchup"]: g["pnl"] for g in status["pnl_by_game"]}
    assert by_game == {
        "MIA @ NY": pytest.approx(88.0 - 80.0 - sc.FEE_NY),
        "DEN @ BOS": pytest.approx(-45.0 - sc.FEE_DEN),
    }
    assert pytest.approx(-42.57) == sc.PNL


def test_an_away_position_is_marked_at_one_minus_the_ask(db: psycopg.Connection) -> None:
    game = {"yes_bid": 0.44, "yes_ask": 0.46}
    assert tools._mark(game, "away", None) == pytest.approx(0.54)
    assert tools._mark(game, "away", False) == 1.0  # settled: away won
    assert tools._mark({"yes_bid": None, "yes_ask": None}, "home", None) is None


def test_a_closed_round_trip_keeps_its_pnl(db: psycopg.Connection, run: int) -> None:
    order = Order(sc.T0 + 3550, sc.NYK, "home", -200, "take profit")
    order_id = live_store.record_order(db, run, order)
    fee = kalshi_taker_fee(0.44, 200)
    live_store.record_fill(db, order_id, Fill(order, sc.T0 + 3555, 0.44, fee))
    status = tools.run_status(db, run)
    assert [p["team"] for p in status["positions"]] == ["DEN"]  # NY is flat now
    by_game = {g["matchup"]: g["pnl"] for g in status["pnl_by_game"]}
    assert by_game["MIA @ NY"] == pytest.approx(88.0 - 80.0 - sc.FEE_NY - fee)
    assert status["pnl"] == pytest.approx(sc.PNL - fee)
    # A flat position needs no price: its P&L stays known when the book disappears.
    db.execute("update game_state set yes_bid = null, yes_ask = null where game_id = %s", [sc.NYK])
    assert tools.run_status(db, run)["pnl"] == pytest.approx(sc.PNL - fee)


def test_pnl_is_withheld_while_a_position_has_no_price(db: psycopg.Connection, run: int) -> None:
    db.execute("update game_state set yes_bid = null, yes_ask = null where game_id = %s", [sc.NYK])
    status = tools.run_status(db, run)
    assert status["pnl"] is None and "no price" in status["pnl_note"]


def test_games_show_score_clock_and_price(db: psycopg.Connection) -> None:
    sc.seed(db)
    board = {g["matchup"]: g for g in tools.games(db, sc.NOW)}
    assert board["MIA @ NY"] | {"updated_minutes_ago": None} == {
        "game_id": sc.NYK,
        "matchup": "MIA @ NY",
        "score": "MIA 72, NY 70",
        "clock": "Q3 4:12",  # the timeout at 4:12 is the latest play
        "status": "live",
        "home_win_price": 0.45,
        "last_play": "NY full timeout",
        "updated_minutes_ago": None,
    }
    assert (board["DEN @ BOS"]["status"], board["DEN @ BOS"]["score"]) == (
        "final",
        "DEN 104, BOS 110",
    )


def test_orders_carry_reasons_and_filter_by_game(db: psycopg.Connection, run: int) -> None:
    newest, ny_buy, den_buy = tools.orders(db, run, sc.NOW)
    assert (newest["status"], newest["error"]) == (
        "rejected",
        "order at ts 1795003016 is before the last event",
    )
    assert ny_buy | {"at": None} == {
        "order_id": ny_buy["order_id"],
        "at": None,
        "minutes_ago": 9.8,  # ordered at T0 + 3015, NOW is T0 + 3600
        "matchup": "MIA @ NY",
        "action": "buy",
        "team": "NY",
        "contracts": 200,
        "reason": sc.NY_REASON,
        "meta": {"run_net": 10, "margin": 2, "timeout": True},
        "status": "filled",
        "fill_price": 0.40,
        "fee": sc.FEE_NY,
        "error": None,
    }
    assert den_buy["team"] == "DEN"
    assert [o["team"] for o in tools.orders(db, run, sc.NOW, game_id=sc.BOS)] == ["DEN"]
    assert len(tools.orders(db, run, sc.NOW, limit=1)) == 1


def test_recent_plays_are_newest_first_with_teams(db: psycopg.Connection, run: int) -> None:
    plays = tools.recent_plays(db, run, sc.NOW, sc.NYK, limit=2)
    assert [(p["clock"], p["score"], p["team"], p["text"]) for p in plays] == [
        ("Q3 4:12", "MIA 72, NY 70", "NY", "NY full timeout"),
        ("Q3 4:30", "MIA 72, NY 70", "MIA", "MIA 3-pointer"),
    ]
    with pytest.raises(ValueError, match="no game"):
        tools.recent_plays(db, run, sc.NOW, "nope")


def test_call_dispatches_and_returns_json(db: psycopg.Connection, run: int) -> None:
    out = json.loads(tools.call(db, run, "orders", {"game_id": sc.BOS, "limit": None}, sc.NOW))
    assert [o["team"] for o in out] == ["DEN"]
    with pytest.raises(ValueError, match="unknown tool"):
        tools.call(db, run, "place_order", {}, sc.NOW)


def test_tool_schemas_are_strict_and_list_every_property() -> None:
    for tool in tools.TOOLS:
        schema = tool["input_schema"]
        assert tool["strict"] is True and schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def test_read_only_connection_refuses_writes(db: psycopg.Connection, run: int) -> None:
    tools.make_read_only(db)
    assert tools.latest_run_id(db) == run
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        db.execute("delete from orders")


def test_clock_text() -> None:
    assert tools.clock_text(2, 61.4) == "Q2 1:01"
    assert tools.clock_text(5, 300) == "OT1 5:00"
    assert tools.clock_text(None, None) == "not started"
