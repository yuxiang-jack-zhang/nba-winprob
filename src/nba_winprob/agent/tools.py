"""Read-only tools the catch-up agent calls: what the bot holds, why, and how games are going.

Every number (P&L, marks, fees) is computed here, so the model reports figures and never
does the arithmetic itself. Results are plain JSON-able dicts. The tools only read, and
``make_read_only`` makes the connection refuse writes as well.

Marks: a settled position is worth $1 or $0 per contract; an open one is marked at what
selling it would fetch now (home: the YES bid; away: 1 - the YES ask).
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from typing import Any

import psycopg
from psycopg.rows import dict_row

from nba_winprob.db import live_store


def make_read_only(conn: psycopg.Connection) -> None:
    """Every later transaction on ``conn`` is read-only (autocommit statements included)."""
    conn.execute("set default_transaction_read_only = on")


def latest_run_id(conn: psycopg.Connection) -> int | None:
    row = conn.execute("select max(id) from runs").fetchone()
    return None if row is None or row[0] is None else int(row[0])


def clock_text(period: int | None, clock: float | None) -> str:
    if period is None or clock is None:
        return "not started"
    minutes, seconds = divmod(round(clock), 60)
    quarter = f"Q{period}" if period <= 4 else f"OT{period - 4}"
    return f"{quarter} {minutes}:{seconds:02d}"


def _board(conn: psycopg.Connection) -> dict[str, dict[str, Any]]:
    return {g["game_id"]: g for g in live_store.game_states(conn)}


def _matchup(board: dict[str, dict[str, Any]], game_id: str) -> str:
    g = board.get(game_id)
    return f"{g['away']} @ {g['home']}" if g else game_id


def _team(board: dict[str, dict[str, Any]], game_id: str, side: str) -> str:
    g = board.get(game_id)
    return str(g[side]) if g else side


def _when(ts: float, now: float) -> dict[str, Any]:
    utc = dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    return {"at": utc, "minutes_ago": round((now - ts) / 60, 1)}


def _settlements(conn: psycopg.Connection, run_id: int) -> dict[str, bool]:
    rows = conn.execute(
        "select game_id, (payload->>'home_won')::boolean from events"
        " where run_id = %s and kind = 'settle'",  # the feed settles each game once
        [run_id],
    ).fetchall()
    return {game_id: bool(home_won) for game_id, home_won in rows}


def _mark(game: dict[str, Any] | None, side: str, home_won: bool | None) -> float | None:
    if home_won is not None:
        return 1.0 if (side == "home") == home_won else 0.0
    if game is None or game["yes_bid"] is None or game["yes_ask"] is None:
        return None
    return float(game["yes_bid"]) if side == "home" else 1.0 - float(game["yes_ask"])


def run_status(conn: psycopg.Connection, run_id: int) -> dict[str, Any]:
    """The run, its positions marked to market, and its P&L."""
    with conn.cursor(row_factory=dict_row) as cur:
        run = cur.execute("select * from runs where id = %s", [run_id]).fetchone()
        fills = cur.execute(
            "select o.game_id, o.team, f.contracts, f.price, f.fee from fills f"
            " join orders o on o.id = f.order_id where o.run_id = %s",
            [run_id],
        ).fetchall()
    if run is None:
        raise ValueError(f"no run {run_id}")
    board, settled = _board(conn), _settlements(conn, run_id)
    cash = -sum(f["contracts"] * f["price"] + f["fee"] for f in fills)
    held: dict[tuple[str, str], dict[str, float]] = {}
    for f in fills:
        h = held.setdefault((f["game_id"], f["team"]), {"contracts": 0, "cost": 0.0})
        h["contracts"] += f["contracts"]
        h["cost"] += f["contracts"] * f["price"] + f["fee"]
    positions, value, unmarked = [], 0.0, False
    for (game_id, side), h in sorted(held.items()):
        if h["contracts"] == 0:
            continue  # closed out; its result is already in cash
        mark = _mark(board.get(game_id), side, settled.get(game_id))
        unmarked |= mark is None
        value += h["contracts"] * (mark or 0.0)
        positions.append(
            {
                "game_id": game_id,
                "matchup": _matchup(board, game_id),
                "team": _team(board, game_id, side),
                "contracts": int(h["contracts"]),
                "net_cost": round(h["cost"], 2),  # fills plus fees, on this game and side
                "mark": mark,
                "value": None if mark is None else round(h["contracts"] * mark, 2),
                "state": (("won" if mark == 1.0 else "lost") if game_id in settled else "open"),
            }
        )
    return {
        "run_id": run["id"],
        "strategy": run["strategy"],
        "broker": run["broker"],  # paper: simulated fills against Kalshi's real book
        "started": run["started_at"].isoformat(),
        "stopped": run["stopped_at"].isoformat() if run["stopped_at"] else None,
        "stop_reason": run["stop_reason"],
        "params": run["params"],
        "positions": positions,
        "fees_paid": round(sum(f["fee"] for f in fills), 2),
        "pnl": None if unmarked else round(cash + value, 2),
        "pnl_note": "a position has no price yet" if unmarked else "fees included",
    }


def games(conn: psycopg.Connection, now: float) -> list[dict[str, Any]]:
    """Every game on the board: score, clock, status and the market's home-win price."""
    out = []
    for g in live_store.game_states(conn):
        book = g["yes_bid"] is not None and g["yes_ask"] is not None
        out.append(
            {
                "game_id": g["game_id"],
                "matchup": f"{g['away']} @ {g['home']}",
                "score": (
                    f"{g['away']} {g['away_score']}, {g['home']} {g['home_score']}"
                    if g["home_score"] is not None
                    else None
                ),
                "clock": clock_text(g["period"], g["clock"]),
                "status": g["status"],
                "home_win_price": round((g["yes_bid"] + g["yes_ask"]) / 2, 3) if book else None,
                "last_play": g["last_play"],
                "updated_minutes_ago": round((now - g["updated_at"].timestamp()) / 60, 1),
            }
        )
    return out


def orders(
    conn: psycopg.Connection,
    run_id: int,
    now: float,
    game_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Newest first, each with the strategy's reason and inputs (``meta``)."""
    board = _board(conn)
    rows = live_store.recent_orders(conn, run_id, 500)
    rows = [o for o in rows if game_id is None or o["game_id"] == game_id][: limit or 20]
    return [
        {
            "order_id": o["id"],
            **_when(o["ts"], now),
            "matchup": _matchup(board, o["game_id"]),
            "action": "buy" if o["contracts"] > 0 else "sell",
            "team": _team(board, o["game_id"], o["team"]),
            "contracts": abs(o["contracts"]),
            "reason": o["reason"],
            "meta": o["meta"],
            "status": o["status"],
            "fill_price": None if o["fill_price"] is None else round(o["fill_price"], 4),
            "fee": None if o["fee"] is None else round(o["fee"], 2),
            "error": o["error"],
        }
        for o in rows
    ]


def recent_plays(
    conn: psycopg.Connection, run_id: int, now: float, game_id: str, limit: int | None = None
) -> list[dict[str, Any]]:
    """The game's latest plays as the bot received them, newest first."""
    board = _board(conn)
    if game_id not in board:
        raise ValueError(f"no game {game_id}; call `games` for the ids")
    g = board[game_id]
    rows = conn.execute(
        "select ts, payload from events where run_id = %s and game_id = %s and kind = 'play'"
        " order by seq desc limit %s",
        [run_id, game_id, limit or 15],
    ).fetchall()
    return [
        {
            **_when(ts, now),
            "clock": clock_text(p["period"], p["clock"]),
            "score": f"{g['away']} {p['away_score']}, {g['home']} {p['home_score']}",
            "team": g.get(p["team"], p["team"]),
            "text": p["text"],
        }
        for ts, p in rows
    ]


def _schema(**props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


_GAME_ID = {"type": ["string", "null"], "description": "ESPN game id from `games`; null for all"}
_LIMIT = {"type": ["integer", "null"], "description": "How many to return; null for default"}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "run_status",
        "description": "The trading run being caught up on: strategy, whether it is still"
        " running, open and settled positions marked to market, fees and total P&L in dollars.",
        "input_schema": _schema(),
    },
    {
        "name": "games",
        "description": "Every game the bot is watching: score, clock, status (scheduled, live,"
        " final), the market's home-win price (0-1) and the last play.",
        "input_schema": _schema(),
    },
    {
        "name": "orders",
        "description": "The bot's orders, newest first, each with its reason and the strategy's"
        " inputs (meta: run size, margin, timeout...), fill price, fee and any error.",
        "input_schema": _schema(game_id=_GAME_ID, limit=_LIMIT),
    },
    {
        "name": "recent_plays",
        "description": "A game's latest plays, newest first, with clock and score after each."
        " Use it for scoring runs, timeouts and how a game is trending.",
        "input_schema": _schema(
            game_id={"type": "string", "description": "ESPN game id from `games`"},
            limit=_LIMIT,
        ),
    },
]
for _tool in TOOLS:
    _tool["strict"] = True


def call(conn: psycopg.Connection, run_id: int, name: str, args: dict[str, Any], now: float) -> str:
    """Run one tool and return its result as JSON. Raises ``ValueError`` for bad input."""
    handlers: dict[str, Callable[[], Any]] = {
        "run_status": lambda: run_status(conn, run_id),
        "games": lambda: games(conn, now),
        "orders": lambda: orders(conn, run_id, now, args.get("game_id"), args.get("limit")),
        "recent_plays": lambda: recent_plays(
            conn, run_id, now, str(args.get("game_id")), args.get("limit")
        ),
    }
    if name not in handlers:
        raise ValueError(f"unknown tool {name}")
    return json.dumps(handlers[name](), default=str)
