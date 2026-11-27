"""Postgres: migrations, the base-table ETL, and the feature store.

These run against a real Postgres (``make db-up``). Each test gets a freshly
created database, so nothing depends on what an earlier run left behind. With no
server they skip locally, and fail when ``NBA_REQUIRE_DB=1`` (set in CI), since a
suite that silently skips proves nothing.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import psycopg
import pytest
from psycopg import errors
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb

from nba_winprob.config import Settings
from nba_winprob.data.pregame_features import (
    build_pregame_master,
    clean_base_tables,
    read_base_csvs,
)
from nba_winprob.db import migrate, store
from tests.conftest import create_database, require_baseline


@pytest.fixture(scope="session")
def loaded_db(settings: Settings) -> Iterator[psycopg.Connection]:
    """One database with the full baseline loaded, shared by the read-only tests."""
    require_baseline(settings.baseline_dir / "games.csv")
    require_baseline(settings.baseline_dir / "team_games.csv")
    for conn in create_database(settings):
        migrate.apply_migrations(conn)
        store.load_baseline(conn, settings)
        yield conn


@pytest.fixture(scope="session")
def raw_tables(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_baseline(settings.baseline_dir / "games.csv")
    require_baseline(settings.baseline_dir / "team_games.csv")
    return read_base_csvs(
        settings.baseline_dir / "games.csv", settings.baseline_dir / "team_games.csv"
    )


def _sample(raw: tuple[pd.DataFrame, pd.DataFrame], n_games: int = 3) -> tuple[pd.DataFrame, ...]:
    games, team_games = raw
    games = games[games["homeWin"].notna()].head(n_games).copy()
    return games, team_games[team_games["gameId"].isin(games["gameId"])].copy()


def _count(conn: psycopg.Connection, table: str) -> int:
    row = conn.execute(f"select count(*) from {table}").fetchone()
    assert row is not None
    return int(row[0])


# --------------------------------------------------------------------------- migrations


def test_migrations_apply_once(empty_db: psycopg.Connection) -> None:
    first = migrate.apply_migrations(empty_db)
    assert first == [p.stem for p in sorted(migrate.MIGRATIONS_DIR.glob("*.sql"))]
    assert migrate.apply_migrations(empty_db) == []
    for table in ("games", "team_games", "pregame_features"):
        assert _count(empty_db, table) == 0


def test_failed_migration_rolls_back_alone(empty_db: psycopg.Connection, tmp_path: Path) -> None:
    (tmp_path / "0001_good.sql").write_text("create table good (id int);")
    (tmp_path / "0002_bad.sql").write_text("create table half_done (id int); select 1/0;")

    with pytest.raises(errors.DivisionByZero):
        migrate.apply_migrations(empty_db, tmp_path)

    recorded = [r[0] for r in empty_db.execute("select version from schema_migrations")]
    assert recorded == ["0001_good"]
    assert _count(empty_db, "good") == 0
    with pytest.raises(errors.UndefinedTable):
        empty_db.execute("select 1 from half_done")


def test_concurrent_runs_apply_each_migration_once(
    empty_db: psycopg.Connection, settings: Settings
) -> None:
    """Two deploys starting together must not both run a migration."""
    info = empty_db.info
    url = make_conninfo(settings.database_url, dbname=info.dbname)
    start, results, failures = threading.Barrier(4), [], []

    def run() -> None:
        with psycopg.connect(url, autocommit=True) as conn:
            start.wait()
            try:
                results.append(migrate.apply_migrations(conn))
            except psycopg.Error as exc:
                failures.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert failures == []
    assert sorted(len(r) for r in results) == [
        0,
        0,
        0,
        len(list(migrate.MIGRATIONS_DIR.glob("*.sql"))),
    ]


# --------------------------------------------------------------------------- schema


def test_schema_rejects_impossible_rows(db: psycopg.Connection) -> None:
    with pytest.raises(errors.CheckViolation):
        db.execute(
            "insert into games (game_id, season, home_team_id, away_team_id) values (1, 2023, 7, 7)"
        )
    db.execute(
        "insert into games (game_id, season, home_team_id, away_team_id) values (1, 2023, 7, 8)"
    )
    with pytest.raises(errors.CheckViolation):
        db.execute(
            "insert into pregame_features (game_id, variant, features) values (1, 'other', %s)",
            [Jsonb({"x": 1.0})],
        )
    with pytest.raises(errors.ForeignKeyViolation):
        db.execute(
            "insert into pregame_features (game_id, variant, features) values (2, 'notebook', %s)",
            [Jsonb({"x": 1.0})],
        )


# --------------------------------------------------------------------------- ETL


def test_upsert_is_idempotent_and_updates(
    db: psycopg.Connection, raw_tables: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    games, team_games = _sample(raw_tables)
    store.upsert_base_tables(db, games, team_games)
    store.upsert_base_tables(db, games, team_games)
    assert _count(db, "games") == len(games)
    assert _count(db, "team_games") == len(team_games)

    first = int(team_games["gameId"].iloc[0])
    changed = team_games.copy()
    changed.loc[changed["gameId"] == first, "num_shots"] = 999.0
    store.upsert_base_tables(db, games, changed)
    shots = db.execute("select num_shots from team_games where game_id = %s", [first]).fetchall()
    assert [r[0] for r in shots] == [999.0, 999.0]
    assert _count(db, "team_games") == len(team_games)


def test_inconsistent_team_game_is_rejected_and_nothing_written(
    db: psycopg.Connection, raw_tables: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    games, team_games = _sample(raw_tables)
    bad = team_games.copy()
    bad.iloc[0, bad.columns.get_loc("homeTeamId")] = 1  # disagrees with games.csv

    with pytest.raises(ValueError, match="disagree"):
        store.upsert_base_tables(db, games, bad)
    assert _count(db, "games") == 0
    assert _count(db, "team_games") == 0


def test_base_tables_round_trip(
    loaded_db: psycopg.Connection,
    raw_tables: tuple[pd.DataFrame, pd.DataFrame],
    base_tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    # Every scheduled game is stored, including the unplayed 2024 ones.
    assert _count(loaded_db, "games") == len(raw_tables[0]) == 9407
    assert _count(loaded_db, "games where home_win is null") == 1229

    got_games, got_tg = clean_base_tables(*store.read_base_tables(loaded_db))
    want_games, want_tg = base_tables
    pd.testing.assert_frame_equal(
        got_games.reset_index(drop=True),
        want_games.sort_values("gameId").reset_index(drop=True),
        check_exact=True,
    )
    # Same columns; order differs (the table groups the game's facts after the stats).
    assert set(got_tg.columns) == set(want_tg.columns)
    pd.testing.assert_frame_equal(
        got_tg[want_tg.columns].reset_index(drop=True),
        want_tg.sort_values(["gameId", "teamId"]).reset_index(drop=True),
        check_exact=True,
    )


@pytest.mark.parametrize("variant", ["notebook", "corrected"])
def test_stored_features_equal_the_csv_build(
    loaded_db: psycopg.Connection,
    base_tables: tuple[pd.DataFrame, pd.DataFrame],
    settings: Settings,
    variant: str,
) -> None:
    """Features rebuilt from the database must be bit-identical to the CSV path."""
    _, want = build_pregame_master(
        *base_tables,
        k=settings.elo_k,
        init=settings.elo_init,
        variant="notebook" if variant == "notebook" else "corrected",
    )
    got = store.read_features(loaded_db, variant)

    assert set(got.columns) == set(want.columns)
    pd.testing.assert_frame_equal(
        got[want.columns].reset_index(drop=True),
        want.sort_values("gameId").reset_index(drop=True),
        check_exact=True,
    )
