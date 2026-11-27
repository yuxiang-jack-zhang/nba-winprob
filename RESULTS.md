# Results

Everything here is produced by `nba-winprob build-features` and
`nba-winprob train-pregame`, seeded and reproducible. Numbers are on the held-out
**2023 season** (1,227 games), which no model trains on.

Status: the **pregame** half is ported and reproduced, a Kalshi paper-trade
backtest of injury and fatigue features is in [section 6](#6-paper-trading-against-kalshi-do-injury-and-fatigue-features-help),
the pre-registered in-game run-reversal backtest is in [section 7](#7-in-game-does-the-kalshi-price-overreact-to-scoring-runs), and the in-game model in [section 8](#8-in-game-does-anything-predict-the-outcome-beyond-the-kalshi-price). The class project's **in-game model**
(logistic + Monte Carlo hybrid over 3.2M possessions) is not yet ported — see
[Not done yet](#not-done-yet).

---

## 1. Feature reproduction: exact

`pregame_master` rebuilt from `games.csv` + `team_games.csv` matches the class
project's committed table **exactly** — 8,162 rows × 79 columns, every value
within 1e-9, identical column order.

```
nba-winprob build-features
# matches baseline exactly (8162 rows x 79 cols)
```

Dataset reality check, which the original write-up was vague about: `games.csv`
holds 9,407 rows but only **8,178 have an outcome label**, spanning **2017–2023**
(7 seasons), not the 1996–2024 range of the raw Kaggle dump. After dropping games
where a team has no prior history, 8,162 remain. The home team wins 56.26%.

## 2. Model reproduction

| Model | Metric | Published | Reproduced | Δ |
|---|---|---|---|---|
| Logistic regression | log-loss | 0.6251 | **0.62505** | −0.00005 |
| | accuracy | 0.6381 | **0.63814** | +0.00004 |
| | AUC | 0.6989 | **0.69893** | +0.00003 |
| | Brier | 0.2180 | **0.21801** | +0.00001 |
| | ECE | 0.0274 | **0.02738** | −0.00002 |
| MLP (128-64) | log-loss | 0.62447 | **0.62443** | −0.00004 |
| | accuracy | 0.64466 | **0.64548** | +0.00082 |
| | AUC | 0.70339 | **0.70428** | +0.00089 |
| | Brier | 0.21745 | **0.21736** | −0.00009 |
| | ECE | 0.03948 | **0.02569** | −0.01379 |
| XGBoost | log-loss | 0.636263 | **0.63486** | −0.00140 |
| | accuracy | 0.639772 | **0.64466** | +0.00489 |
| | AUC | 0.696014 | **0.69759** | +0.00158 |
| | Brier | 0.222609 | **0.22198** | −0.00063 |
| | ECE | 0.051546 | **0.05583** | +0.00428 |

All within the stated tolerance of 0.005 log-loss and 0.01 accuracy. Why the
three differ in how closely they land:

- **Logistic regression** matches to five decimals. It is a convex fit on a fixed
  split, so anything other than a near-exact match would have meant a port bug.
- **MLP** matches on log-loss, accuracy, AUC and Brier, but its **ECE is
  noticeably better** (0.0257 vs 0.0395). The original run seeded nothing — not
  torch, not the shuffled batch order — so its exact weights are unrecoverable.
  ECE is the most fragile metric here: 10 equal-width bins over 1,227 games
  leaves the tail bins with a handful of games each, so it moves far more than
  log-loss does under the same noise. This run is seeded and reproduces itself
  bit-for-bit.
- **XGBoost** is deterministic, and the split and `scale_pos_weight`
  (0.7761519302615193) match the original exactly, so the residual gap is the
  library: xgboost 3.4.1 bins features and breaks ties differently from the 2.x
  that produced the published numbers.

Split sizes match the notebooks exactly, which is what confirms the data is
being carved up identically: grouped split 4,506 / 2,429 / 1,227; chronological
split 5,705 / 1,230 / 1,227.

## 3. The published comparison was not apples to apples

The original table presents three models as if trained the same way. They were
not, and the difference is large enough to matter:

| | Logistic + MLP | XGBoost |
|---|---|---|
| Features | **76** (home, away and diff, plus all context) | **43** (23 `*_diff` + 20 context) |
| Validation | `GroupShuffleSplit`, whole seasons, `random_state=42` | the penultimate season (2022) |
| Train rows | 4,506 | 5,705 |

XGBoost also trains on 27% more rows and validates on a single season. Its 43
columns include **five duplicated columns** — `rest_diff`, `b2b_diff`,
`season_winrate_diff`, `home_home_vs_away_away_wr_diff` and `elo_diff` all end
in `_diff`, so the suffix match picked them up and the explicit context list
added them again. Both recipes are preserved exactly as written, because
"fixing" either changes the numbers this table is meant to reproduce.

So "the MLP beat XGBoost" is a statement about two different pipelines, not two
model classes on equal footing.

## 4. Two feature bugs, found and fixed

Porting surfaced two defects in the feature code. Both are preserved under
`--variant notebook` and fixed under `--variant corrected`.

**Bug 1 — rest and back-to-backs measured within one side only.**
`days_since_last_home` was the gap since the team's previous *home* game, because
the helper only ever saw one side's rows. Since real back-to-backs are usually
home-then-away or away-then-home, the flags almost never fired:

| | `is_b2b_home` | `is_b2b_away` |
|---|---|---|
| Original | 7.6% | 13.0% |
| Corrected | 14.3% | 20.2% |
| True NBA rate | ≈19% per team | ≈19% per team |

**Bug 2 — the home/away win-rate split was half placeholder.** The home-only
rate was computed on home rows and the away-only rate on away rows, then both
were filled with 0.5. Every row therefore carried one real rate and one constant.
Carried into `pregame_master`, **four of the 76 features were dead**:

| Feature | Original | Corrected |
|---|---|---|
| `home_away_winrate` | constant 0.5 (1 distinct value) | 346 distinct values |
| `away_home_winrate` | constant 0.5 (1 distinct value) | 399 distinct values |
| `home_home_minus_away_wr` | exactly `home_home_winrate − 0.5` | a real home/away split |
| `away_home_minus_away_wr` | exactly `0.5 − away_away_winrate` | a real home/away split |

Two carried no information; two were affine copies of columns already present.

### What the fix was worth

| Model | log-loss | accuracy | AUC | Brier | ECE |
|---|---|---|---|---|---|
| Logistic, original | 0.62505 | 0.63814 | 0.69893 | 0.21801 | 0.02738 |
| Logistic, corrected | **0.62273** | **0.64629** | **0.70496** | **0.21678** | 0.03294 |
| MLP, original | 0.62443 | 0.64548 | 0.70428 | 0.21736 | 0.02569 |
| MLP, corrected | **0.62238** | **0.65281** | **0.70734** | **0.21641** | 0.02873 |
| XGBoost, original | 0.63486 | 0.64466 | 0.69759 | 0.22198 | 0.05583 |
| XGBoost, corrected | **0.63447** | 0.64466 | **0.70114** | **0.22169** | 0.05797 |

Log-loss, accuracy, AUC and Brier improve for all three. **ECE gets slightly
worse across the board** (+0.002 to +0.006): the corrected models are sharper,
and sharper probabilities are harder to keep calibrated. Reported rather than
buried, since a win-probability service is judged on calibration — if that
trade-off matters more than discrimination, the fix needs a calibration layer on
top, not a rollback.

### The headline

> Fixing two feature bugs bought **3.7× more log-loss** than replacing logistic
> regression with a neural network did.

- Logistic → MLP, same features: **0.00062** log-loss
- Fixing the features, same logistic regression: **0.00232** log-loss

And the best corrected MLP (0.62238) beats the best corrected logistic
regression (0.62273) by 0.00035 — about a third of a thousandth of a nat. On this
feature set, **model capacity is nearly irrelevant and feature quality is not**.
That is the project's real finding, and it only became visible once the notebook
logic was in code that could be re-run under changed inputs.

## 5. Leakage

A pregame feature must not see the game it describes. An earlier milestone
version of this project leaked target-game statistics; the rebuild fixed it, and
this port now proves it rather than asserting it.

The decisive test reverses one game's outcome, rebuilds every feature, and
asserts that game's own features are bit-identical — the label is the only thing
a pregame feature could leak. Run over 40 sampled games, plus:

- a **control** confirming a flip *does* change later games, so the invariance
  test cannot pass by the features ignoring outcomes altogether;
- independent recomputation of the rolling-5 and season-to-date win rates from
  strictly-preceding games, for 40 sampled (game, side) pairs;
- the same invariance check against the `corrected` code path, since that is
  where a new leak would hide.

**Mutation-checked.** Removing the `.shift()` that makes `season_winrate`
leak-free causes the test to fail on **10 out of 10** sampled games. A leakage
test that has never been shown to fail is not evidence.

## 6. Paper trading against Kalshi: do injury and fatigue features help?

A backtest, not live trading: `papertrade/backtest.py` replays **1,559 settled
Kalshi NBA game markets** (April 2025 – November 2026) at the last one-minute
quote before tip-off, crossing the spread and paying the taker fee. No orders
were placed. Every game is predicted by a model fitted only on earlier games
(refits at 2025-04-15, 2025-10-01 and 2026-10-01).

Two model families, each fitted with and without the new feature blocks:

- **Lean** — logistic regression on carry-over Elo, win rates and recent margin.
- **XGBoost** — this project's recipe (`xgb_feature_columns`, `xgb_params`, the
  corrected feature pipeline) **retrained on ESPN data**. The saved artifact
  cannot be used: its inputs come from a Kaggle archive that stops in 2024.
  Shot zones are by distance from ESPN coordinates, so the features match the
  originals in spirit, not value for value. Its own rest and back-to-back
  columns are removed from "base" so rest enters only through the fatigue block.

The blocks: **fatigue** is rest days, back-to-back, and games in the last 7 and
14 days; **injury** is the share of a team's usual minutes (previous 10 games)
held by players who did not play.

### Forecast quality

Change in log-loss against each family's own base, paired by game, with 95%
bootstrap intervals. Negative is better.

| Added | Lean | XGBoost |
|---|---|---|
| Fatigue | +0.0006 [−0.0039, +0.0053] | +0.0007 [−0.0014, +0.0028] |
| Injury | **−0.0059 [−0.0111, −0.0005]** | **−0.0036 [−0.0070, −0.0003]** |
| Both | −0.0052 [−0.0119, +0.0018] | −0.0026 [−0.0065, +0.0012] |

**Injury features help; fatigue features do not.** The injury gain is
significant in both families. Fatigue adds nothing on its own and dilutes the
injury gain enough that the combined block is no longer significant.

### Against the market

Flat $100 on a side whenever the model beats cost plus fee by 3 cents.

| | Log-loss | Bets | P&L | ROI | 95% CI of P&L |
|---|---|---|---|---|---|
| Kalshi mid | **0.5747** | | | | |
| Lean base | 0.6022 | 956 | −$14,346 | −15.0% | [−23,114, −5,614] |
| Lean + injury | 0.5963 | 912 | −$12,519 | −13.7% | [−21,485, −3,547] |
| Lean + both | 0.5970 | 899 | −$8,886 | −9.9% | [−17,478, +190] |
| XGBoost base | 0.6287 | 1,185 | −$14,612 | −12.3% | [−25,646, −3,094] |
| XGBoost + both | 0.6261 | 1,166 | −$16,331 | −14.0% | [−27,471, −4,674] |

**No model beats the market, and every one loses money.** The market is 0.02
log-loss ahead of the best model. It already prices injuries: the gap between
Kalshi's price and the lean base model correlates −0.48 with the injury feature,
so the feature moves the model toward the market rather than past it. The lean
logistic model beats the XGBoost recipe throughout, consistent with section 4.

### Playoffs

Over 173 playoff and play-in games the lean model out-forecast the market
(0.643 vs 0.654) and made about +$2,200, but the interval is [−$1,100, +$5,700].
Suggestive, not a result.

The 2026 Finals (Knicks beat Spurs 4–1) had the textbook fatigue setup in Game 1:
Spurs on 4 days' rest with 6 games in 14 days, Knicks on 9 days' rest with 3.
The models profited (+$523 for lean + both) by fading the Spurs at home, where
the market had them at 63–68% and they lost all three. **The fatigue features
were not the reason.** The base model already rated the teams closer than the
market, and the new blocks moved the Spurs' probability *up* in Games 1 and 2
because the Knicks had more minutes missing. Five games cannot settle it.

### Caveats

- **Hindsight in the injury feature.** It uses who actually sat, so a
  last-minute scratch counts as known. The pre-tip quote limits this, but the
  injury models still have a slight edge a live trader would not.
- 51 games have no ESPN box score and are treated as "no injury information";
  53 matched Kalshi games are dropped for lack of team history.
- The fee is per contract without Kalshi's round-up to a cent per order.
- Fills are assumed at the quoted price for $100; no depth or slippage.

## 7. In-game: does the Kalshi price overreact to scoring runs?

A backtest, not live trading. The idea comes from my manual in-game trading: in a
close game, a 10–15 point run tends to be followed by a counter-run, especially
after a timeout. The question tested is whether the **market price** overreacts
to runs, measured as what a trader would actually have made.

**Pre-registered.** The rule, its exits, latency and splits were written into the
[design doc](docs/superpowers/specs/2026-11-26-intragame-trading-design.md) before
any in-game data was examined, and this is the one run of that spec. Variants are
reported alongside the primary result, never instead of it.

- **Data:** 1,570 Kalshi NBA game markets (April 2025 – November 2026) with ESPN
  play-by-play and Kalshi one-minute candles. 42 more games were rejected by the
  loader: ESPN timestamps shifted by hours or a day (both 2026 DST changeovers and
  Oct 28–30, 2025), play-by-play that does not sum to the final score, or missing
  summaries.
- **Rule (v1):** over the trailing 4:00 of game clock, one team outscored the
  other by 10–15; the game is within 9; more than 2:00 left. Buy ~$100 of the team
  that was run on. Exit at +5¢ on the mid, after 6:00 of game clock, or at 2:00
  left, selling at the bid.
- **Execution:** fills at the first one-minute candle close at least 30 s after the
  trigger play, crossing the spread, paying Kalshi's taker fee.

### Result: no edge

| Run | Trades | P&L | P&L/trade | ROI | 95% CI of P&L |
|---|---|---|---|---|---|
| **v1, latency 30 s (primary)** | 1,712 | −$17,382 | −$10.15 | −11.3% | [−$20,053, −$14,663] |
| v1, latency 0 s | 1,719 | −$18,086 | −$10.52 | −11.7% | [−$20,769, −$15,476] |
| v1, latency 60 s | 1,709 | −$16,743 | −$9.80 | −11.0% | [−$19,526, −$13,915] |
| v1, latency 120 s | 1,692 | −$15,529 | −$9.18 | −10.3% | [−$18,715, −$12,130] |
| Hold to settlement | 1,067 | −$12,470 | −$11.69 | −12.9% | [−$20,073, −$4,781] |

Every split loses: with or without a timeout by the run-on team (−10.7% and −8.7%),
first half and second half, regular season and playoffs.

**The price does not revert.** Measured without any exit rule, the mid of the side
bought moves −0.10¢, +0.11¢, −0.19¢ and −0.35¢ at 1, 3, 5 and 10 minutes after
entry. Every 95% interval spans zero and is within ±1¢. Waiting longer (latency
120 s) loses slightly *less*, the opposite of what an overreaction that fades
would produce.

**Where the money goes.** Of $153k staked, taker fees took 7.1% and the spread
(1.3¢ on an average price of 45¢) about 2.9%. Price movement after entry
contributed about −1.4%, within noise. A round trip with taker orders costs
roughly **10% of the stake**, so an in-game strategy needs a large, reliable
price move just to break even.

### What this does and does not show

- At one-minute resolution, after a run is visible in the play-by-play, Kalshi's
  in-game price has already absorbed it. The rule as written loses its costs.
- It does **not** test reacting within seconds while watching a live broadcast;
  one-minute candles cannot. It also cannot see the context I weigh by hand
  (matchups, who is on the floor, fatigue), which is the next step's job.
- The cost structure is the binding constraint. Any in-game strategy here must
  either find moves well over 10% of the stake or cut costs: resting limit orders
  instead of crossing the spread, and fewer, larger-edge trades.

## 8. In-game: does anything predict the outcome beyond the Kalshi price?

The follow-up to section 7, also pre-registered: the features, models and
evaluation were committed in
[a prereg](docs/superpowers/specs/2026-11-27-ingame-model-prereg.md) before any
result was computed. 177,385 one-minute snapshots from 1,569 games, using only
plays at least 30 s older than each price (the assumed feed lag). Walk-forward by
game date: train on earlier games, test on the next 15%, four times. Intervals
resample games.

**Who wins** (104,311 test snapshots, lower is better):

| Forecast | Log-loss | Brier |
|---|---|---|
| **Kalshi mid, as is** | **0.48739** | **0.16143** |
| Logistic recalibration of the mid | 0.48994 | 0.16197 |
| Logistic: mid + margin, time, runs, timeouts, pregame price, price momentum | 0.49189 | 0.16264 |
| Gradient boosting, same features | 0.58384 | 0.19023 |

Adding in-game features makes the forecast **worse** than the raw price: +0.0045
log-loss, 95% CI [+0.0017, +0.0071]. No split (half, close vs not close) shows
a gain. Gradient boosting overfits badly, most likely by memorising past games
through near-unique pregame prices, since the effective sample is games, not
snapshots.

**The next 3 minutes of price:** a ridge model explains 0.1% of the variance. Its
largest predicted move is 1.6¢, while a taker round trip costs 3.6¢ on average,
so it never trades. Gradient boosting's larger predictions are noise and lose
3.1¢ per contract after costs (CI [−4.5¢, −1.7¢]).

**Reading:** at one-minute resolution, Kalshi's in-game NBA price already
contains the score, the clock, runs, timeouts and the pregame line. What is left
is far smaller than the cost of trading. Kalshi's NBA game markets also charge
**maker** fees (`quadratic_with_maker_fees`, apparently 0.0175·C·P·(1−P) per
filled resting order versus 0.07 for takers; coefficient not yet confirmed from
the official PDF). Resting orders cut costs by roughly 4×, but they do not
create an edge where these models find none.

## Not done yet

- **The in-game model.** The notebook reads `combined/pbpstats_rg_all.csv`, which
  exists on no machine here, and never saved its 3.2M-row intermediate, so the
  hybrid cannot be reproduced without re-running ETL over the 13 GB archive.
  Checkpoints to hit are recorded: 3,748,379 possessions from 2017 → 3,704,053
  after the `games.csv` join → 3,221,319 with a resolvable defending team over
  8,176 games, split 2,256,201 / 965,118. Target metrics: accuracy 0.761,
  log-loss 0.470, AUC 0.845, Brier 0.157.
- **The known hybrid bug** is not fixed yet: `HybridWinProbPredictor.predict`
  calls `logistic_fn(score_diff, time_remaining)` with two arguments, so Elo,
  season win rate and back-to-back silently default to 0 on every blended
  prediction. Fixing it will move the published in-game numbers, possibly down
  at some stages, and both sets will be reported here.
- **No trained model from the original survives.** `*.pkl` was gitignored and
  `ingame_model.pkl` is on no disk, so all four models are retrained from source.
- The feature-bug fixes are **not yet carried into the in-game model**, which
  consumes `b2b_diff` and `season_winrate_diff` from this same table.
