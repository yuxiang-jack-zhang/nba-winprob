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
**Decision:** `COPY` into a temp table, then `insert … on conflict do update … where` the row
differs, all in one transaction per load (base tables and both feature variants commit
together or not at all). Re-running is a no-op: identical rows are not rewritten or re-stamped,
and the counts report rows changed. Each feature variant is replaced as a whole, so features of
a game that dropped out of the build are deleted. **Why:** idempotent and safe to re-run on a
live database. **Limit:** a game or team game removed from the source is not removed from the
base tables. The baseline is append-only, so this does not arise yet. **Revisit if:** a source
starts retracting games.

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

## D10. In-game data: wallclock order, re-added scores, reject rather than repair — 2026-11-27
**Context:** ESPN play-by-play for 1,604 games; its list order, per-play scores and some
timestamps turned out to be unreliable.
**Decision:** replay plays by wallclock, then `sequenceNumber` (ESPN lists by game clock, which
misorders plays within a stoppage, such as an and-one free throw before the basket). Rebuild
scores by summing `scoreValue` in replay order; ESPN's per-play score fields go stale after
late-entered plays (1,802 plays in 196 games). Reject a game if its plays do not sum to ESPN's
final score (3 games) or its wallclocks are shifted: first play outside [tip − 10 min,
tip + 50 min], or a > 50-minute jump between plays (31 games, clustered on both 2026 DST
changeovers and Oct 28–30, 2025). Drop Kalshi quotes after the final buzzer. Cache raw
responses whole, gzipped. **Why:** each check is one cheap invariant that catches a whole
class of error; repairing shifted clocks would mean guessing which plays moved, for 2% of
games. **Revisit if:** the rejected games cluster in a way that biases results (they are
mostly early regular season), or ESPN's alternate plays API is needed for the 8 missing
playoff summaries.

## D11. Simulated fills: strictly later quote, latency, round trips — 2026-11-27
**Decision:** an order fills at the first quote with `ts >= order.ts + latency` *and*
`ts > order.ts`, so even at zero latency it cannot fill at the quote it reacted to.
`submit` rejects an order stamped earlier than the current event (no backdating past the
latency). Oversells and buys outside (0, 1) raise at fill time. At settlement, pending orders
are recorded as cancelled. A game ending with open positions and no settlement is an
error, not a silent valuation. Results are one row per round trip (flat → flat or
settlement), carrying the entry order's reason and meta plus the closing order's meta as
`exit_<key>`. The bootstrap resamples every replayed game, untraded ones as 0. **Why:** each
rule closes a way a backtest can quietly look better than live trading would be; all 21
targeted mutations of the broker were caught by the tests.

## D12. RunReversal v1: how the written spec was turned into code — 2026-11-27
**Decision:** windows are open at the start and closed at the end. The runner is whichever
side is ahead over the window, not the scorer, since the spec defines a run by net points.
"Within 30 s after the trigger" is wallclock. Only team timeouts count. One trade per game,
and a pending entry counts as open. Re-arm only after an actual entry. Exit size is what is
held. Because the full timeout flag includes 30 s *after* the trigger, it is computed on the
exit order (`exit_timeout_by_run_on_team`), and splits use that. The "hold to settlement"
variant needed the 2:00 exit separated from the 2:00 entry cutoff (`exit_seconds_left`, also
120 by default, so v1 is unchanged). **Why:** these are interpretations, not tuning, and each
is pinned by a test (28 mutations caught).

## D13. Result: v1 has no edge, and costs are the binding constraint — 2026-11-27
**Context:** one run of the pre-registered spec over 1,570 games (RESULTS.md section 7).
**Finding:** −11.3% ROI, 95% CI of P&L entirely negative, at every latency and in every split.
The price of the side bought does not revert (mean move within ±0.4¢ at 1–10 minutes).
Taker fees (7.1% of stake) plus spread (~2.9%) account for nearly all of the loss.
**Decision:** v1 is not traded, and it is not tuned on these results either. Next: the
planned model phase asks whether any in-game feature predicts outcomes *beyond the market
price*. Any later strategy is costed with maker (resting) orders as well as taker orders,
since a 10% round-trip cost rules out small edges. **Revisit if:** sub-minute data becomes
available. Reacting within seconds of a run was never tested here.

## D14. In-game model result: nothing beyond the price — 2026-11-27
**Context:** phase 3, pre-registered (`docs/superpowers/specs/2026-11-27-ingame-model-prereg.md`).
**Finding:** the raw Kalshi mid out-forecasts every model, including logistic recalibration
of itself; in-game features make forecasts worse (RESULTS.md section 8). Price-move models
never clear costs. Walk-forward cuts are moved to day boundaries so no model trains on a game
from a test game's date. Gradient boosting's built-in early stopping holds out snapshots, not
games, which is a likely cause of its overfitting; noted, not fixed, since fixing it after
seeing the result would be tuning. **Decision:** no in-game strategy built on these features.
The engine and live stack stay, for strategies with a different information source.
**Revisit if:** a new information source appears that the market might be slow to price
(sub-minute data, lineup and injury news within the game).

## D15. Live paper trading: real book, receipt time, forward-only board — 2026-11-27
**Decision:** paper trading is `SimBroker` (latency 0) fed live quotes polled from Kalshi's
public production book, so it fills at the next polled quote, about one poll interval
later. Every live event is stamped with receipt time: ESPN wallclock lags the Kalshi quotes
already processed, and decisions can only happen once data arrives. The runner never
re-sorts a poll. `game_state` only moves forward in game time, so a late-inserted play cannot
roll the board back. Broker refusals become `rejected` orders, not crashes. Every stop
(Ctrl-C, `data/live/STOP`, a time limit, a crash) closes the run with its reason and cancels
unfilled orders. The CLI offers `paper` and `demo` only; no code path places real-money
orders. Quotes for a game start only after its play backlog has gone out, so a strategy
cannot trade on stale plays replayed at start-up. **Limits:** postponed games never settle (the feed emits no `Settle`), and ESPN
corrections to plays already emitted are not re-sent.

## D16. Kalshi demo broker: V2 orders, reduce-only sells, a host guard — 2026-11-27
**Context:** Kalshi retired `POST /portfolio/orders` (410 Gone) for V2 single-book orders,
and now issues Ed25519 API keys by default.
**Decision:** orders go to `POST /portfolio/events/orders` as immediate-or-cancel limit
orders priced to cross (`bid` at the YES ask to buy home, `ask` at the YES bid to buy away),
with sells `reduce_only` so they can never open the other side. Both Ed25519 and RSA-PSS
signing are supported. The client refuses any host but Kalshi's two demo hosts unless
`allow_live=True`, which nothing passes. The transport never retries, since a resent POST can
double an order. The demo exchange lists the same NBA tickers as production, but its books
are fake and thin, so it rehearses the order path only. **Revisit if:** phase 5 (real orders)
starts: that needs its own caps, kill switch and an explicit decision by Jack.

## D17. Catch-up agent: local tool loop, code does the arithmetic — 2026-11-27
**Context:** the agent answers "what is my bot holding, why, and how are the games going?"
from the live tables.
**Options:** Managed Agents (hosted loop, but its sandbox cannot reach a local Postgres);
a model-written-SQL tool (flexible, but the model would add up P&L itself); a fixed set of
read-only tools on a plain Messages API loop.
**Decision:** four fixed read-only tools (`run_status`, `games`, `orders`,
`recent_plays`) and a hand-written loop with the client injected, so tests script it. The
tools compute P&L, marks and fees: open positions are marked at what selling would fetch now
(home at the YES bid, away at 1 − the YES ask), settled ones at $1/$0, and P&L is withheld
when any position has no price rather than counted at zero. The CLI's connection is set
`default_transaction_read_only`. Evals are deterministic regex checks against a seeded
scenario whose numbers the tool tests pin, opt-in because they bill. **Revisit if:** Jack
wants push updates (Discord) or follow-up questions, which need a session and a schedule.

