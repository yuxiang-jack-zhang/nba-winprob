# Decision log

Each entry: **context → options (pros/cons) → decision → why → revisit if**. Newest at the
bottom. Decisions from milestone 1 (season splits, LFS baseline, rebuilding features, both
variants, no pickles, torch threads) are summarised in the README's
[Design decisions](../README.md#design-decisions).

---

## D1. Postgres through psycopg 3 and plain SQL, no ORM — 2026-11-26
**Context:** milestone 2 puts games, team games and features in Postgres; the API (milestone 3)
reads features by game and writes a prediction log.
**Options:** *SQLAlchemy ORM* — + familiar models; − a second description of every table, and
bulk loads end up bypassing it anyway. *SQLAlchemy Core* — + composable queries; − a large
dependency for about six queries. *psycopg 3 + SQL* — + `COPY` for bulk loads, `sql.Identifier`
for safe composition, the schema reads as the SQL it is; − no query builder.
**Decision:** psycopg 3 and SQL. **Why:** a handful of queries over three tables; the SQL is
the clearest description of them. **Revisit if:** the API grows dynamic filtering that would
mean building SQL strings by hand.

## D2. Migrations: numbered SQL files and a small runner, not Alembic — 2026-11-26
**Context:** the schema must be applied the same way locally, in CI and on deploy.
**Options:** *Alembic* — + standard, autogenerate; − autogenerate needs ORM models (see D1), so
it would just wrap hand-written SQL in Python files. *Numbered `.sql` files + runner* — + the
migration is the SQL; ~40 lines to own.
**Decision:** `src/nba_winprob/db/migrations/NNNN_name.sql`, applied by `db/migrate.py`, each in
its own transaction with its `schema_migrations` row, under a Postgres advisory lock. No down
migrations: roll forward with a new file. **Why:** keeps D1's "the SQL is the source"; the lock
and per-file transaction are the only parts that need care, and both are tested
(`test_concurrent_runs_apply_each_migration_once` caught a real race in the first draft:
`create table if not exists` is not safe under concurrency without the lock). **Revisit if:**
migrations need data backfills written in Python.

## D3. Features stored as JSONB per (game, variant) — 2026-11-26
**Context:** 76 feature columns, two variants, and models that use different subsets (MLP 76,
XGBoost 43).
**Options:** *76 typed columns* — + types enforced; − a migration whenever a feature changes,
and a second table or column set per variant. *JSONB object per row* — + feature changes need
no migration; serving picks columns by the `feature_cols` list saved with each artifact; −
values are not type-checked by Postgres.
**Decision:** `pregame_features(game_id, variant, features jsonb)`. **Why:** the artifact
already records exactly which columns it needs; the database only has to hold them faithfully,
which `test_stored_features_equal_the_csv_build` checks bit-for-bit. JSON cannot hold NaN, and
none of the 8,162 rows contain one; Postgres rejects the write if one ever appears.
**Revisit if:** analytics queries over individual features become common (add generated columns).

## D4. The DB is what features are built from, read back in a fixed order — 2026-11-26
**Context:** milestone 1 rebuilt features from the CSVs. With a database, either it stores
features computed from the CSVs, or features are computed from what it stores.
**Decision:** `db load` loads the CSVs, reads them back, and builds both variants from that.
**Why:** it proves the stored base tables are sufficient and faithful, rather than assuming it.
**Finding:** the pipeline is not fully order-independent. Two 2023 games have no date, and
the stable sorts on `gameDate` keep their input order, so a shuffled input changes five
win-rate features on those two games by up to 0.014. Reading in `(game_id, team_id)` order
matches the CSV build exactly; reversing that order makes the test fail. Not fixed in the
feature code, since fixing it would move the reproduced numbers. **Revisit if:** the
reproduction constraint is dropped; then break date ties by `gameId` in the pipeline itself.

## D5. `team_games` stores each fact once; the loader checks the CSV's copies — 2026-11-26
**Context:** `team_games.csv` repeats home team, away team, season, date and result from
`games.csv`.
**Decision:** the table keeps only `(game_id, team_id)` and the shot stats; the repeated columns
come from a join on read. On load, every repeated value must equal the one in `games`, and the
team must be one of the two that played, or the whole load is rejected (`ValueError`, nothing
written). **Why:** two copies of a fact can drift; checking them once at the trust boundary
costs a merge. All 16,356 baseline rows pass.

## D6. Loads are upserts, not truncate-and-reload — 2026-11-26
**Context:** the prediction log (milestone 3) will reference games, so wiping and reloading
games would break or cascade into it.
**Decision:** `COPY` into a temp table, then `insert … on conflict do update`, one transaction
per load. Re-running is a no-op; changed values are updated. **Why:** idempotent and safe to
re-run on a live database. **Limit:** a row removed from the source is not removed from the
table. The baseline is append-only, so this does not arise yet. **Revisit if:** a source starts
retracting games.

## D7. DB tests run against real Postgres, one fresh database per test — 2026-11-26
**Context:** the constraints, the upsert and the migration lock are Postgres behaviour; a mock
or SQLite would test something else.
**Decision:** `tests/test_db.py` creates a uniquely named database per test (one shared, loaded
database for the read-only tests) and drops it afterwards. With no server the tests skip
locally; with `NBA_REQUIRE_DB=1`, set in CI, they fail instead. CI runs a `postgres:16` service.
**Why:** the same rule as the LFS check: a suite that silently skips in CI proves nothing.
Every test was mutation-checked: each targeted break (read order, upsert, consistency check,
check constraint, migration lock) fails its test.

## D8. Scope of the milestone 2 schema — 2026-11-26
**Context:** the roadmap listed games, features, game states and predictions.
**Decision:** ship `games`, `team_games`, `pregame_features`. **Deferred:** `predictions` and a
`model_versions` registry go in milestone 3 with the API that writes them, as migration 0002.
`game_states` waits for the possession data, which needs the 13 GB archive (RESULTS.md, "Not
done yet"). All 9,407 scheduled games are loaded, including the 1,229 unplayed 2024 games with
`home_win` null; they have no team-game rows, so no features yet. **Why:** tables arrive with
the code that uses them, so every table is exercised by a test the day it exists.

## D9. Direction: an in-game Kalshi trading system, not a prediction API — 2026-11-26
**Context:** milestones 3–7 were a FastAPI prediction service, an LLM agent, evals and an AWS
deploy. Jack trades Kalshi NBA game markets in-game by hand and wants the repo to be the
system he trades with: paper first, then real orders in live games.
**Options:** *keep the API roadmap* — + already planned; − a pregame-probability API does not
help in-game trading. *Pivot to trading* — + the work serves a real use; + the backtest is an
honest, falsifiable result either way; − more moving parts (live data, an exchange API, risk).
**Decision:** pivot (design: `docs/superpowers/specs/2026-11-26-intragame-trading-design.md`).
Keep Postgres and the LLM agent, repurposed as a catch-up agent over the bot's positions and
trades. The API and cloud deploy become "where the live trader runs".
**Why:** in-game prices move on information a pregame model cannot see; the question worth
answering is whether the market overreacts to runs. **Revisit if:** the pre-registered
backtest and the model both find nothing beyond the market price; then the engine is still
the base for other strategies, but live trading of this one is off.
