"""Apply the numbered SQL files in ``migrations/`` in order, each exactly once.

Each file runs in its own transaction together with its ``schema_migrations``
row, so a failing migration leaves no trace and the ones before it stay applied.
An advisory lock makes concurrent runs (two deploys starting at once) safe.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_LOCK_ID = 229_2025  # arbitrary, fixed: identifies this project's migration lock


def apply_migrations(conn: psycopg.Connection, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply pending migrations and return the versions applied, in order."""
    with conn.transaction():
        # Locked too: concurrent "create table if not exists" races on the catalog.
        conn.execute("select pg_advisory_xact_lock(%s)", [_LOCK_ID])
        conn.execute(
            "create table if not exists schema_migrations ("
            " version text primary key,"
            " applied_at timestamptz not null default now())"
        )
    applied = []
    for path in sorted(directory.glob("*.sql")):
        with conn.transaction():
            conn.execute("select pg_advisory_xact_lock(%s)", [_LOCK_ID])
            done = conn.execute(
                "select 1 from schema_migrations where version = %s", [path.stem]
            ).fetchone()
            if done:
                continue
            # Bytes: psycopg's types want a literal string, and a file's contents are not one.
            # Sent without parameters, a file of several statements runs as one script.
            conn.execute(path.read_text(encoding="utf-8").encode())
            conn.execute("insert into schema_migrations (version) values (%s)", [path.stem])
        applied.append(path.stem)
    return applied
