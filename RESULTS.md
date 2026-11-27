# Results

Everything here is produced by `nba-winprob build-features` and
`nba-winprob train-pregame`, seeded and reproducible. Numbers are on the held-out
**2023 season** (1,227 games), which no model trains on.

Status: the **pregame** half is ported and reproduced. The **in-game** half
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
