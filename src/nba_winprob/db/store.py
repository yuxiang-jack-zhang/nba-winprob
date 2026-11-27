"""Load the base tables into Postgres, read them back, and store pregame features.

The database is the source the features are built from: ``load_baseline`` loads
the CSVs, reads them back, rebuilds both feature variants from what was stored,
and writes those. Reading back in ``(game_id, team_id)`` order matters: the
feature pipeline breaks date ties by input order (two games have no date), and
this order reproduces the CSV build bit-for-bit.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from nba_winprob.config import Settings
from nba_winprob.data.pregame_features import (
    FeatureVariant,
    build_pregame_master,
    clean_base_tables,
    read_base_csvs,
)
from nba_winprob.models.pregame import ID_COLS

# CSV header -> column, in the CSV's own order so a read-back frame lines up with it.
GAME_COLUMNS = {
    "gameId": "game_id",
    "awayTeamId": "away_team_id",
    "homeTeamId": "home_team_id",
    "season": "season",
    "homeWin": "home_win",
    "gameDate": "game_date",
}
# team_games.csv repeats these from games.csv; the table stores them once.
_FROM_GAMES = ["homeTeamId", "awayTeamId", "homeWin", "season", "gameDate"]
VARIANTS: tuple[FeatureVariant, ...] = ("notebook", "corrected")


def connect(settings: Settings) -> psycopg.Connection:
    """An autocommit connection: every write here opens an explicit transaction."""
    return psycopg.connect(settings.database_url, autocommit=True)


def load_baseline(conn: psycopg.Connection, settings: Settings) -> dict[str, int]:
    """Load the baseline CSVs, then build and store both feature variants from the DB.

    One transaction: if any step fails, nothing is written, so the base tables and
    both variants always come from the same load. ``features_*`` count rows changed.
    """
    raw = read_base_csvs(
        settings.baseline_dir / "games.csv", settings.baseline_dir / "team_games.csv"
    )
    with conn.transaction():
        upsert_base_tables(conn, *raw)
        games, team_games = clean_base_tables(*read_base_tables(conn))
        counts = {"labelled_games": len(games), "team_games": len(team_games)}
        for variant in VARIANTS:
            _, master = build_pregame_master(
                games, team_games, k=settings.elo_k, init=settings.elo_init, variant=variant
            )
            counts[f"features_{variant}"] = write_features(conn, master, variant)
    return counts


def upsert_base_tables(
    conn: psycopg.Connection, games: pd.DataFrame, team_games: pd.DataFrame
) -> None:
    """Insert or update raw ``games`` and ``team_games`` (CSV-shaped) in one transaction.

    Raises ``ValueError``, writing nothing, if a key repeats, a team-game row
    disagrees with its game about who is home, the season, the date or the result,
    or any team-game row (incoming or already stored) names a team that did not play.
    """
    _check_consistent(games, team_games)
    stats = [c for c in team_games.columns if c not in ("gameId", "teamId", *_FROM_GAMES)]

    game_rows = games[list(GAME_COLUMNS)].copy()
    game_rows["homeWin"] = game_rows["homeWin"].astype("boolean")  # raises unless 0, 1 or missing
    game_rows["gameDate"] = pd.to_datetime(game_rows["gameDate"]).dt.date

    with conn.transaction():
        _upsert(conn, "games", list(GAME_COLUMNS.values()), ["game_id"], _records(game_rows))
        _upsert(
            conn,
            "team_games",
            ["game_id", "team_id", *stats],
            ["game_id", "team_id"],
            _records(team_games[["gameId", "teamId", *stats]]),
        )
        # Stored rows too: changing a game's teams must not orphan its old team rows.
        row = conn.execute(
            "select count(*) from team_games t join games g using (game_id)"
            " where t.team_id not in (g.home_team_id, g.away_team_id)"
        ).fetchone()
        if row and row[0]:
            raise ValueError(f"{row[0]} stored team_games rows name a team that did not play")


def _check_consistent(games: pd.DataFrame, team_games: pd.DataFrame) -> None:
    for name, df, key in (
        ("games", games, ["gameId"]),
        ("team_games", team_games, ["gameId", "teamId"]),
    ):
        if df.duplicated(key).any():
            raise ValueError(f"{name} has duplicate {key} rows")
    merged = team_games.merge(
        games[["gameId", *_FROM_GAMES]], on="gameId", how="left", suffixes=("", "_g")
    )
    if merged["homeTeamId_g"].isna().any():
        raise ValueError("team_games has rows for games missing from games")

    bad = (merged["teamId"] != merged["homeTeamId_g"]) & (
        merged["teamId"] != merged["awayTeamId_g"]
    )
    for col in _FROM_GAMES:
        mine, theirs = merged[col], merged[f"{col}_g"]
        if col == "gameDate":
            mine, theirs = pd.to_datetime(mine), pd.to_datetime(theirs)
        bad |= (mine != theirs) & ~(mine.isna() & theirs.isna())
    if bad.any():
        sample = merged.loc[bad, ["gameId", "teamId"]].head(3).to_dict("records")
        raise ValueError(f"{int(bad.sum())} team_games rows disagree with games, e.g. {sample}")


def read_base_tables(conn: psycopg.Connection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every game and team game with the CSVs' columns and dtypes, for ``clean_base_tables``."""
    games = _frame(
        conn.execute(
            sql.SQL("select {} from games g order by game_id").format(_game_select(GAME_COLUMNS))
        ),
        list(GAME_COLUMNS),
    )
    # Every other team_games column is a stat, named exactly as in the CSV.
    stats = [c for c in _columns(conn, "team_games") if c not in ("game_id", "team_id")]
    cur = conn.execute(
        sql.SQL(
            "select t.game_id, t.team_id, {stats}, {game} from team_games t"
            " join games g using (game_id) order by t.game_id, t.team_id"
        ).format(
            stats=sql.SQL(", ").join(sql.Identifier("t", c) for c in stats),
            game=_game_select(_FROM_GAMES),
        )
    )
    team_games = _frame(cur, ["gameId", "teamId", *stats, *_FROM_GAMES])
    return _with_csv_types(games), _with_csv_types(team_games)


def write_features(conn: psycopg.Connection, master: pd.DataFrame, variant: str) -> int:
    """Store one row per game: its feature columns as a JSON object; returns rows changed.

    ``master`` replaces the variant: rows for games it no longer has are deleted.
    """
    feature_cols = [c for c in master.columns if c not in ID_COLS]
    values = master[feature_cols].to_numpy(dtype=float)
    rows = (
        (int(game_id), variant, Jsonb(dict(zip(feature_cols, map(float, row), strict=True))))
        for game_id, row in zip(master["gameId"], values, strict=True)
    )
    with conn.transaction():
        conn.execute(
            "delete from pregame_features where variant = %s and not (game_id = any(%s))",
            [variant, [int(g) for g in master["gameId"]]],
        )
        return _upsert(
            conn,
            "pregame_features",
            ["game_id", "variant", "features"],
            ["game_id", "variant"],
            rows,
            refresh=["built_at"],
        )


def read_features(conn: psycopg.Connection, variant: str) -> pd.DataFrame:
    """``gameId``, ``homeWin``, ``season`` and the stored features for one variant."""
    cur = conn.execute(
        "select f.game_id, g.home_win, g.season, f.features::text from pregame_features f"
        " join games g using (game_id) where f.variant = %s order by f.game_id",
        [variant],
    )
    rows = cur.fetchall()
    ids = pd.DataFrame([r[:3] for r in rows], columns=["gameId", "homeWin", "season"]).astype(
        {"gameId": "int64", "homeWin": "float64", "season": "int64"}
    )
    # Parse the JSON text with the stdlib so floats round-trip exactly.
    features = pd.DataFrame([json.loads(r[3]) for r in rows], dtype="float64")
    return pd.concat([ids, features], axis=1)


# --------------------------------------------------------------------------- helpers


def _upsert(
    conn: psycopg.Connection,
    table: str,
    columns: Sequence[str],
    key: Sequence[str],
    rows: Iterable[Sequence[Any]],
    refresh: Sequence[str] = (),
) -> int:
    """COPY ``rows`` into a staging table, then insert-or-update by ``key``; returns
    rows inserted or changed (an identical row is left alone).

    ``refresh`` names defaulted columns (timestamps) reset when a row changes.
    Must run inside a transaction. The stage is dropped here, not on commit: inside an
    outer transaction this one is only a savepoint, and the next call reuses the name.
    """
    stage = sql.Identifier(f"_stage_{table}")
    cols = sql.SQL(", ").join(map(sql.Identifier, columns))
    values = [c for c in columns if c not in key]
    updates = values + list(refresh)
    conn.execute(
        sql.SQL("create temp table {} (like {} including defaults) on commit drop").format(
            stage, sql.Identifier(table)
        )
    )
    with conn.cursor().copy(sql.SQL("copy {} ({}) from stdin").format(stage, cols)) as copy:
        for row in rows:
            copy.write_row(row)
    cur = conn.execute(
        sql.SQL(
            "insert into {table} ({cols}) select {cols} from {stage}"
            " on conflict ({key}) do update set {updates}"
            " where {mine} is distinct from {theirs}"
        ).format(
            table=sql.Identifier(table),
            cols=cols,
            stage=stage,
            key=sql.SQL(", ").join(map(sql.Identifier, key)),
            updates=sql.SQL(", ").join(
                sql.SQL("{c} = excluded.{c}").format(c=sql.Identifier(c)) for c in updates
            ),
            # row(...) so a single column still compares as a row
            mine=sql.SQL("row({})").format(
                sql.SQL(", ").join(sql.Identifier(table, c) for c in values)
            ),
            theirs=sql.SQL("row({})").format(
                sql.SQL(", ").join(sql.Identifier("excluded", c) for c in values)
            ),
        )
    )
    conn.execute(sql.SQL("drop table {}").format(stage))
    return cur.rowcount


def _records(df: pd.DataFrame) -> list[list[Any]]:
    """Rows as plain Python values with ``None`` for missing; psycopg does not adapt numpy."""
    return df.astype(object).where(df.notna(), None).to_numpy().tolist()


def _columns(conn: psycopg.Connection, table: str) -> list[str]:
    cur = conn.execute(
        "select column_name from information_schema.columns"
        " where table_schema = current_schema() and table_name = %s order by ordinal_position",
        [table],
    )
    return [r[0] for r in cur]


def _game_select(headers: Iterable[str]) -> sql.Composed:
    """Columns of ``games g`` named by CSV header; the date as ISO text, as the CSV has it."""
    parts = []
    for header in headers:
        column = sql.Identifier("g", GAME_COLUMNS[header])
        expr = sql.SQL("{}::text").format(column) if header == "gameDate" else column
        parts.append(sql.SQL("{} as {}").format(expr, sql.Identifier(header)))
    return sql.SQL(", ").join(parts)


def _frame(cur: psycopg.Cursor[Any], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(cur.fetchall(), columns=columns)


def _with_csv_types(df: pd.DataFrame) -> pd.DataFrame:
    """The dtypes ``pd.read_csv`` gives the baseline files: float stats and labels, int IDs."""
    types = {
        c: (
            "int64"
            if c in ("gameId", "teamId", "homeTeamId", "awayTeamId", "season")
            else "float64"
        )
        for c in df.columns
        if c != "gameDate"
    }
    return df.astype(types)
