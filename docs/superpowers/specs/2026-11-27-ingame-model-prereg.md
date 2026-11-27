# In-game model (phase 3) — pre-registration

2026-11-27. Written and committed before any evaluation was run. Run once; the
results are reported as they come out, whatever they are.

**Question.** Does any in-game information predict the outcome *beyond the Kalshi
price*? If the market price already contains it, that is the result.

## Spec

- **Snapshots:** one per in-game Quote with 0.03 < mid < 0.97 and > 120 s of game time
  left (regulation or OT), mid = (bid+ask)/2 of home YES. Features use only plays with
  `ts <= quote.ts - 30` (the 30 s feed lag assumed in v1).
- **Features:** logit(mid); margin (home − away); game seconds left;
  margin / sqrt(seconds_left + 60); home run net over trailing 240 s and over trailing
  120 s of game clock (home pts − away pts); home timeout and away timeout in the
  trailing 60 s of game clock (bools); pregame logit(mid) (last quote at or before the
  first play); period; 1-minute and 3-minute change in mid (from earlier quotes;
  momentum of the price itself).
- **Target A (outcome):** home_won. Models: (A0) logistic on logit(mid) only; (A1)
  logistic on all features (standardized); (A2) sklearn HistGradientBoostingClassifier
  on all features, default params except random_state=0 and max_iter=200. Also report
  raw mid as a forecast (no model).
- **Target B (price move):** home mid 3 minutes later minus mid now (first quote at or
  after +180 s; drop if none before settlement). Models: (B1) ridge (alpha=1) on all
  features; (B2) HistGradientBoostingRegressor, random_state=0.
- **Evaluation:** walk-forward by game date. Sort games by tip time; first 40% of games
  = initial train; then 4 test blocks of 15% each, each trained on all games before it
  (expanding). Metrics pooled over test snapshots: log-loss and Brier (target A),
  out-of-sample R² (target B, vs predicting 0 change). Uncertainty: 95% bootstrap
  intervals resampling GAMES (4000 draws, seed 0) for Δlog-loss of A1/A2 vs A0 and vs
  raw mid, and for R².
- **Economics of B:** for test snapshots where |predicted 3-min move| exceeds the
  round-trip taker cost at that snapshot (spread + Kalshi taker fee on both legs, per
  contract: entry fee 0.07·p·(1−p) at the entry price and the same at the exit price;
  use engine.events.kalshi_taker_fee's formula per contract without the per-order cent
  rounding), report count and mean realized signed move minus cost, with a
  game-bootstrap CI. That is the only "would it make money" number for this phase; no
  engine backtest unless it is clearly positive.
- **Reported splits** (A1 vs A0 Δlog-loss only): first half vs second half;
  |margin| ≤ 9 vs > 9.

## Clarifications (fixed before any result was seen)

- **Game state at a snapshot** is the state after the last play with
  `ts <= quote.ts - 30` (score, period, clock). A quote with no such play is not
  in-game and gives no snapshot. Quotes after the final buzzer are already dropped by
  the loader (D10). Game time left, the 2:00 filter and OT use
  `features.ingame.game_seconds_left` (in OT: seconds left in the current OT period).
- **Trailing windows** are in game seconds elapsed, `(now − w, now]`, where now is the
  game clock of that last play, exactly as `features.ingame.RunTracker` computes them.
  Runs are `RunTracker.run_net("home")`. Timeouts are team timeouts only (official
  timeouts carry no team).
- **Pregame mid** is clipped to [0.01, 0.99] before the logit (it is not filtered by the
  0.03–0.97 range). A game with no quote at or before its first play gives no snapshots.
- **Mid changes:** mid now minus the mid of the last quote with `ts <= quote.ts − 60`
  (resp. − 180), pregame quotes included. A snapshot with no such quote is dropped.
- **Period** enters as a number (1–4, 5+ for OT).
- **Models:** sklearn `LogisticRegression` defaults (L2, C=1.0) with max_iter=1000; A0
  on raw logit(mid). A1 and B1 standardize all features with a `StandardScaler` fit on
  the training block. HGB models get unscaled features. Target B models are trained and
  scored only on snapshots that have a target.
- **Walk-forward dates:** a game's date is the Kalshi market's `day`. Cut points at
  40/55/70/85% of games in tip order are moved back to the first game of that cut game's
  date, so all games of one date fall in one block and no model trains on a game dated
  at or after any game it is tested on.
- **Bootstrap:** one set of 4000 game resamples (seed 0) of the pooled test games,
  shared by every interval. A pooled statistic is recomputed on each resample from
  per-game sums (Δlog-loss = Σ differences / Σ snapshots; R² = 1 − ΣSSE / Σy²; the
  economics mean = Σ net / Σ trades). Split intervals resample the games in that split.
- **Economics:** direction = sign of the B prediction (buy home YES at the ask if up,
  buy NO at 1 − bid if down). Entry price p = that ask or 1 − bid. Cost per contract =
  spread now + 2 × 0.07·p·(1−p): the exit price is unknown when deciding, so the exit
  fee is taken at p. Realized net per contract = sign × (mid at +3 min − mid now) − that
  same cost. Reported for B1 and B2 separately.
- **Δlog-loss sign:** model minus baseline, so negative means the model is better.
