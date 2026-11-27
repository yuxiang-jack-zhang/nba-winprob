"""What to push to Discord: fills, rejected orders, game recaps, run stops, and digest timing.

The notifier reads the live tables and remembers what it has reported, so each event is
pushed once. It starts from what is already in the tables, so a restart does not replay
the evening. It never calls the model: ``poll`` says when a digest is due and the caller
asks the agent for one. Everything here is templated text, free and instant.
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row

from nba_winprob.agent import tools

DIGEST_EVERY_S = 30 * 60
DISCORD_LIMIT = 2000  # characters per message


def money(x: float | None) -> str:
    return "n/a (no price yet)" if x is None else f"{'+' if x >= 0 else '-'}${abs(x):.2f}"


class Notifier:
    def __init__(
        self, conn: psycopg.Connection, now: float, digest_every_s: float = DIGEST_EVERY_S
    ) -> None:
        self.conn, self.digest_every_s, self.last_digest = conn, digest_every_s, now
        row = conn.execute("select coalesce(max(id), 0) from fills").fetchone()
        self.last_fill = int(row[0]) if row else 0
        self.rejected = self._ids("select id from orders where status = 'rejected'")
        self.stopped = self._ids("select id from runs where stopped_at is not null")
        self.recapped = {(run, game) for run, game, _ in self._settled_games()}

    def poll(self, now: float) -> tuple[list[str], bool]:
        """New messages to push, and whether a digest is due (true at most every interval)."""
        messages = self._fills() + self._rejections() + self._recaps() + self._stops()
        run = tools.latest_run_id(self.conn)
        row = self.conn.execute(
            "select exists (select 1 from runs where id = %s and stopped_at is null)",
            [run],
        ).fetchone()
        live = any(g["status"] == "live" for g in tools._board(self.conn, run).values())
        due = bool(row and row[0]) and live and now - self.last_digest >= self.digest_every_s
        if due:
            self.last_digest = now
        return messages, due

    def _ids(self, sql: str) -> set[int]:
        return {int(r[0]) for r in self.conn.execute(sql).fetchall()}

    def _fills(self) -> list[str]:
        with self.conn.cursor(row_factory=dict_row) as cur:
            rows = cur.execute(
                "select f.id, o.game_id, o.team, f.contracts, f.price, f.fee, o.reason"
                " from fills f join orders o on o.id = f.order_id where f.id > %s order by f.id",
                [self.last_fill],
            ).fetchall()
        board = tools._board(self.conn)
        out = []
        for r in rows:
            self.last_fill = r["id"]
            verb = "Bought" if r["contracts"] > 0 else "Sold"
            out.append(
                f"{verb} {abs(r['contracts'])} {tools._team(board, r['game_id'], r['team'])}"
                f" @ {r['price']:.2f} in {tools._matchup(board, r['game_id'])}"
                f" (fee ${r['fee']:.2f}). Why: {r['reason']}"
            )
        return out

    def _rejections(self) -> list[str]:
        with self.conn.cursor(row_factory=dict_row) as cur:
            rows = cur.execute(
                "select id, game_id, team, contracts, error from orders"
                " where status = 'rejected' and not (id = any(%s)) order by id",
                [list(self.rejected)],
            ).fetchall()
        board = tools._board(self.conn)
        out = []
        for r in rows:
            self.rejected.add(r["id"])
            verb = "buy" if r["contracts"] > 0 else "sell"
            out.append(
                f"Order rejected: {verb} {abs(r['contracts'])}"
                f" {tools._team(board, r['game_id'], r['team'])}"
                f" in {tools._matchup(board, r['game_id'])}: {r['error']}"
            )
        return out

    def _settled_games(self) -> list[tuple[int, str, dict[str, Any]]]:
        """(run, game, that game's P&L entry) for the latest run's traded, settled games."""
        run = tools.latest_run_id(self.conn)
        if run is None:
            return []
        settled = tools._settlements(self.conn, run)
        status = tools.run_status(self.conn, run)
        return [(run, g["game_id"], g) for g in status["pnl_by_game"] if g["game_id"] in settled]

    def _recaps(self) -> list[str]:
        board = tools._board(self.conn)
        out = []
        for run, game_id, entry in self._settled_games():
            if (run, game_id) in self.recapped:
                continue
            self.recapped.add((run, game_id))
            g = board.get(game_id)
            score = (
                f"{g['away']} {g['away_score']}, {g['home']} {g['home_score']}" if g else game_id
            )
            out.append(f"Final: {score}. Bot P&L on this game: {money(entry['pnl'])}.")
        return out

    def _stops(self) -> list[str]:
        with self.conn.cursor(row_factory=dict_row) as cur:
            rows = cur.execute(
                "select id, strategy, broker, stop_reason from runs"
                " where stopped_at is not null and not (id = any(%s)) order by id",
                [list(self.stopped)],
            ).fetchall()
        out = []
        for r in rows:
            self.stopped.add(r["id"])
            pnl = tools.run_status(self.conn, r["id"])["pnl"]
            out.append(
                f"Run {r['id']} ({r['strategy']}, {r['broker']}) stopped: {r['stop_reason']}."
                f" P&L {money(pnl)}."
            )
        return out


def chunks(text: str, limit: int = DISCORD_LIMIT) -> list[str]:
    """Split ``text`` into Discord-sized messages, at line breaks where possible."""
    out, current = [], ""
    for line in text.splitlines(keepends=True):
        for piece in (line[i : i + limit] for i in range(0, len(line), limit)):
            if len(current) + len(piece) > limit:
                out.append(current)
                current = ""
            current += piece
    out.append(current)
    return [c.strip() for c in out if c.strip()]
