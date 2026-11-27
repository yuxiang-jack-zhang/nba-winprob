"""The catch-up agent's read-only tools, against a seeded evening of paper trading."""

from __future__ import annotations

import json

import psycopg
import pytest

from nba_winprob.agent import tools
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
    assert pytest.approx(-42.57) == sc.PNL


def test_an_away_position_is_marked_at_one_minus_the_ask(db: psycopg.Connection) -> None:
    game = {"yes_bid": 0.44, "yes_ask": 0.46}
    assert tools._mark(game, "away", None) == pytest.approx(0.54)
    assert tools._mark(game, "away", False) == 1.0  # settled: away won
    assert tools._mark({"yes_bid": None, "yes_ask": None}, "home", None) is None


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
