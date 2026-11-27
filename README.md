# NBA Win Probability

NBA win-probability models and a trading system for Kalshi's NBA game markets,
aimed at **in-game** trading: prove a strategy on paper against real market
prices, then let it trade live games under hard risk limits. An LLM agent
answers "what is my bot holding, and why?" for when I'm not watching.

**Status: milestone 3 of 8.** Done: the pregame models (ported, reproduced,
tested) and Postgres (migrations, ETL, features that round-trip exactly). In
progress: in-game history, a backtest engine shared with live trading, and the
first strategy — see the [design](docs/superpowers/specs/2026-11-26-intragame-trading-design.md).
**No order has been placed on any exchange.** Nothing here claims an edge until
the backtest in [RESULTS.md](RESULTS.md) says so.

## Origin and credit

This started as a team project for **Stanford CS229 (Autumn 2025)** by
**[Young Kim](https://github.com/youngkim4)**, **Mia Xiao** and
**[Jack Zhang](https://github.com/yuxiang-jack-zhang)**. The original repository
is **[youngkim4/CS229Project](https://github.com/youngkim4/CS229Project)** and
this repo keeps its MIT license.

To be clear about who did what:

| | |
|---|---|
| **From the class project** (joint work) | Feature engineering, the pregame logistic/MLP/XGBoost models, and the in-game logistic + Monte Carlo hybrid — all as Jupyter notebooks. |
| **This repo** (Jack) | Porting that logic into a tested package, making it reproducible, finding and fixing two feature bugs, and building the service, agent, evals and infrastructure around it. |

The model *results* are the team's. The engineering around them is the point of
this repo.

## What's interesting here

- **Exact reproduction, then improvement.** The feature table rebuilds to the
  original byte-for-byte (8,162 × 79, within 1e-9), and only then do the fixes
  land — so every later change has a measured before and after.
- **Two real bugs found in the port.** Four of the 76 features turned out to be
  constant or redundant, and the back-to-back flags fired at less than half the
  true rate. Fixing them bought **3.7× more log-loss than replacing logistic
  regression with a neural network did** — the project's actual finding.
- **Leakage is proven, not asserted.** Reversing a game's outcome must leave that
  game's own features untouched. The test is mutation-checked: introduce a leak
  and it fails on 10/10 sampled games.
- **The database is tested like code.** Migrations, upserts and constraints run
  against real Postgres, every test is mutation-checked, and the concurrency test
  caught a real migration race before it shipped ([decisions](docs/decisions.md)).
- **Honest reporting.** The original comparison of three models turned out to
  use three different feature sets and splits. [RESULTS.md](RESULTS.md) says so.

Full numbers and discussion: **[RESULTS.md](RESULTS.md)**.

## Results at a glance

Held-out 2023 season, 1,227 games. Home team wins 56.26% of the time.

| Model | log-loss | accuracy | AUC | Brier | ECE |
|---|---|---|---|---|---|
| Logistic regression | 0.62505 | 0.63814 | 0.69893 | 0.21801 | **0.02738** |
| MLP (128-64) | 0.62443 | 0.64548 | 0.70428 | 0.21736 | 0.02569 |
| XGBoost | 0.63486 | 0.64466 | 0.69759 | 0.22198 | 0.05583 |
| **MLP, corrected features** | **0.62238** | **0.65281** | **0.70734** | **0.21641** | 0.02873 |

The MLP's edge over logistic regression is about 0.0006 log-loss — roughly a
rounding error. Logistic regression is the best calibrated of the originals, and
XGBoost loses outright. **Feature quality mattered far more than model capacity**,
which is the opposite of what the modelling effort assumed going in.

Predicting NBA games from pregame information is genuinely hard: ~65% accuracy
against a 56% base rate is a real but modest edge, and any claim much above that
from pregame features alone should be treated as a leakage bug until proven
otherwise.

## Quickstart

Requires **Python 3.12**, [uv](https://docs.astral.sh/uv/), and
**[Git LFS](https://git-lfs.com)**.

```bash
git lfs install && git clone https://github.com/yuxiang-jack-zhang/nba-winprob && cd nba-winprob
```

> **Git LFS is required.** The three baseline tables in `data/baseline/` (~16 MB)
> are stored in LFS so the repo reproduces its own numbers without the 1.6 GB
> Kaggle download. Without `git lfs`, you get small text pointer files instead of
> CSVs; every test that needs them **skips with a message** rather than failing
> confusingly. If that happens, run `git lfs pull`.

```bash
uv sync --all-groups
```

macOS only: `brew install libomp`. The xgboost wheel needs an OpenMP runtime; the
logistic and MLP models work without it.

```bash
# Rebuild the feature tables and diff them against the committed baseline
uv run nba-winprob build-features

# Train all three pregame models; writes artifacts/ + metrics
uv run nba-winprob train-pregame --variant notebook
uv run nba-winprob train-pregame --variant corrected
uv run nba-winprob show-metrics

make check    # ruff, mypy, pytest
```

Postgres (Docker): migrate, load the baseline, and build both feature variants
from what was stored. The DB tests skip without it.

```bash
make db       # docker compose up, nba-winprob db migrate, nba-winprob db load
```

## Layout

```
src/nba_winprob/
  config.py                 settings and paths, env-overridable; no absolute paths
  data/pregame_features.py  the feature pipeline (both variants)
  features/                 feature builders shared by training and serving
  models/
    pregame.py              logistic, MLP, XGBoost: splits, training, params
    metrics.py              log-loss, accuracy, AUC, Brier, ECE, by-stage
    artifacts.py            versioned save/load with metrics alongside
  db/
    migrations/             numbered SQL, applied once each under an advisory lock
    migrate.py              the migration runner
    store.py                base-table ETL, consistency checks, feature store
  cli.py                    nba-winprob entry point
tests/                      incl. test_no_leakage.py, the bug-pinning tests, test_db.py
docs/decisions.md           every non-trivial decision: options, choice, why
data/baseline/              the three tables, in Git LFS
data/{raw,interim,processed}/  gitignored
artifacts/                  gitignored
```

## Architecture

Only the left column exists today.

```
          BUILT                             NOT BUILT YET
  ┌──────────────────────────┐    ┌─────────────────────────────┐
  │ nba_winprob package      │    │ FastAPI  /predict/pregame   │
  │  features + models       │    │          /predict/ingame    │
  │  metrics + artifacts     │───▶│          /ask (LLM agent)   │
  │  CLI, tests, CI          │    │ Postgres: predictions,      │
  │ Postgres                 │    │   model_versions,           │
  │  games, team_games,      │    │   game_states               │
  │  pregame_features        │    │ Eval harness (50+ cases)    │
  │  migrations, ETL         │    │ Docker → AWS, IaC, budget   │
  └──────────────────────────┘    └─────────────────────────────┘
```

## Design decisions

**Season-based splits, never shuffled.** The latest season (2023) is the test set
and validation holds whole seasons. Shuffling games would let a model train on
games played after the ones it is scored on, which inflates every metric and is
the standard way sports-prediction results get overstated.

**The baseline tables are committed (via LFS).** The brief said never commit
data. Taken literally, nobody could reproduce the numbers without a 1.6 GB Kaggle
download, which makes "reproducible" a claim rather than a fact. 16 MB of derived
tables in LFS is the better trade; the raw archive and the possession-level data
stay out.

**Rebuild features, don't read them.** Training rebuilds `pregame_master` from
the base tables rather than loading the committed CSV. The committed file is a
baseline to diff against, not an input whose provenance is assumed.

**Both feature variants kept.** `--variant notebook` reproduces the class project
including its bugs; `--variant corrected` fixes them. Keeping both is what makes
the improvement measurable instead of asserted.

**No pickled estimators.** joblib for scikit-learn, `state_dict` for torch. The
original pickled a live model object; that artifact is unloadable today.

**torch is pinned to one thread.** On macOS, torch and xgboost load separate
OpenMP runtimes and the combination crashes the process rather than raising.
Importing xgboost first and limiting torch to one thread both turn out to be
necessary; `tests/test_openmp_coexistence.py` guards it in a subprocess, since a
regression is a SIGSEGV rather than a test failure.

## Limitations

- Pregame data covers **2017–2023 only** (8,162 labelled games), not the raw
  dataset's 1996–2024.
- Features are **shot-profile aggregates plus context** (rest, win rates, Elo).
  There is no player-level information, so injuries, rotations and trades are
  invisible to the model — a large part of why pregame accuracy sits near 65%.
- Elo resets every season, so it measures within-season form, not franchise
  strength.
- The corrected features are **sharper but slightly less calibrated** (ECE +0.002
  to +0.006). A calibration layer is the right fix, and isn't built yet.
- No live data. Everything is historical; `stats.nba.com` blocks cloud IPs.

## Roadmap

| | Milestone | Status |
|---|---|---|
| 1 | Repo, package refactor, reproduced metrics | **done** |
| 2 | Postgres schema, migrations, ETL | **done** (in-game possession table waits for the archive) |
| 3 | In-game history (Kalshi minute prices + ESPN plays), event-driven engine, pre-registered run-reversal backtest | in progress |
| 4 | In-game model: what does the market price get wrong? | |
| 5 | Live paper trading on Kalshi's demo exchange; orders and fills in Postgres | |
| 6 | Live orders: off by default, dollar caps, kill switch | |
| 7 | Catch-up agent (Claude + tools over positions, trades, games) and its evals | |
| 8 | Deploy the live trader (Docker, cloud, budget alert) | |

The roadmap changed after milestone 2, from "prediction API" to "trading
system"; see [D9](docs/decisions.md).

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2025 youngkim4 and contributors.
