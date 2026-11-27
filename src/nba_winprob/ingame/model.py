"""Phase 3: does in-game information predict the outcome beyond the Kalshi price?

Implements the pre-registration in docs/superpowers/specs/2026-11-27-ingame-model-prereg.md,
run once. Snapshots are one per in-game quote; models are fit walk-forward by date.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nba_winprob.config import Settings
from nba_winprob.engine.events import Event, Play, Quote, Settle
from nba_winprob.features.ingame import RunTracker
from nba_winprob.ingame.history import game_index
from nba_winprob.ingame.report import load_games

LAG_S = 30.0  # ESPN's live feed lag, as in v1
HORIZON_S = 180.0
MIN_SECONDS_LEFT = 120.0
MID_LOW, MID_HIGH = 0.03, 0.97
TEST_CUTS = (0.40, 0.55, 0.70, 0.85)
N_BOOT = 4000
FEATURES = [
    "logit_mid",
    "margin",
    "seconds_left",
    "margin_scaled",
    "run_240",
    "run_120",
    "home_timeout",
    "away_timeout",
    "pregame_logit",
    "period",
    "dmid_1m",
    "dmid_3m",
]


def _mid(q: Quote) -> float:
    return (q.yes_bid + q.yes_ask) / 2


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def game_snapshots(events: list[Event]) -> pd.DataFrame:
    """One row per in-game quote that passes the filters: features, targets, bid/ask."""
    plays = [e for e in events if isinstance(e, Play)]
    quotes = [e for e in events if isinstance(e, Quote)]
    home_won = next(e for e in events if isinstance(e, Settle)).home_won
    stamps = [q.ts for q in quotes]
    pre = bisect_right(stamps, plays[0].ts) - 1
    if pre < 0:
        return pd.DataFrame()
    pregame = _logit(min(max(_mid(quotes[pre]), 0.01), 0.99))
    long, short = RunTracker(240), RunTracker(120)
    seen, rows = 0, []
    for q in quotes:
        while seen < len(plays) and plays[seen].ts <= q.ts - LAG_S:
            long.update(plays[seen])
            short.update(plays[seen])
            seen += 1
        mid = _mid(q)
        if seen == 0 or not MID_LOW < mid < MID_HIGH or long.seconds_left <= MIN_SECONDS_LEFT:
            continue
        j1 = bisect_right(stamps, q.ts - 60) - 1
        j3 = bisect_right(stamps, q.ts - 180) - 1
        if j3 < 0:  # j3 <= j1, so this also covers a missing 1-minute quote
            continue
        k = bisect_left(stamps, q.ts + HORIZON_S)
        now = long.elapsed
        rows.append(
            {
                "game_id": q.game_id,
                "ts": q.ts,
                "logit_mid": _logit(mid),
                "margin": long.margin,
                "seconds_left": long.seconds_left,
                "margin_scaled": long.margin / math.sqrt(long.seconds_left + 60),
                "run_240": long.run_net("home"),
                "run_120": short.run_net("home"),
                "home_timeout": long.timeout_called("home", game_window=(now - 60, now)),
                "away_timeout": long.timeout_called("away", game_window=(now - 60, now)),
                "pregame_logit": pregame,
                "period": long.period,
                "dmid_1m": mid - _mid(quotes[j1]),
                "dmid_3m": mid - _mid(quotes[j3]),
                "mid": mid,
                "bid": q.yes_bid,
                "ask": q.yes_ask,
                "home_won": home_won,
                "move": _mid(quotes[k]) - mid if k < len(quotes) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def walk_forward(games: pd.DataFrame) -> list[tuple[pd.Index, pd.Index]]:
    """(train game_ids, test game_ids) per test block, from columns game_id, day, tip_ts.

    Cuts sit at 40/55/70/85% of games in tip order, moved back to the start of the cut
    game's day, so train games are always dated strictly before every test game.
    """
    order = games.sort_values("tip_ts")
    day, ids = order["day"], pd.Index(order["game_id"])
    starts = [day.iloc[int(f * len(order))] for f in TEST_CUTS]
    blocks = []
    for i, start in enumerate(starts):
        test = day >= start
        if i + 1 < len(starts):
            test &= day < starts[i + 1]
        blocks.append((ids[(day < start).to_numpy()], ids[test.to_numpy()]))
    return blocks


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """``test`` with each model's prediction: P(home wins) for A*, 3-minute move for B*."""
    x, xt, y = train[FEATURES].to_numpy(float), test[FEATURES].to_numpy(float), train["home_won"]
    has_b = train["move"].notna().to_numpy()
    out = test.copy()
    a0 = LogisticRegression(max_iter=1000).fit(train[["logit_mid"]].to_numpy(), y)
    out["A0"] = a0.predict_proba(test[["logit_mid"]].to_numpy())[:, 1]
    a1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(x, y)
    out["A1"] = a1.predict_proba(xt)[:, 1]
    a2 = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(x, y)
    out["A2"] = a2.predict_proba(xt)[:, 1]
    move = train["move"].to_numpy()[has_b]
    out["B1"] = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x[has_b], move).predict(xt)
    out["B2"] = HistGradientBoostingRegressor(random_state=0).fit(x[has_b], move).predict(xt)
    return out


def game_ci(frame: pd.DataFrame, num: str, den: str, games: pd.Index) -> tuple[float, float]:
    """95% interval of sum(num) / sum(den), resampling ``games`` (absent ones add 0)."""
    sums = frame.groupby("game_id")[[num, den]].sum().reindex(games, fill_value=0.0)
    idx = np.random.default_rng(0).integers(0, len(sums), size=(N_BOOT, len(sums)))
    ratio = sums[num].to_numpy()[idx].sum(1) / sums[den].to_numpy()[idx].sum(1)
    return float(np.quantile(ratio, 0.025)), float(np.quantile(ratio, 0.975))


def log_loss(p: pd.Series, y: pd.Series) -> pd.Series:
    """Per-snapshot log-loss."""
    p = p.clip(1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def taker_trades(preds: pd.DataFrame, model: str) -> pd.DataFrame:
    """Snapshots where |predicted move| beats the round-trip taker cost, and each net result.

    Cost per contract = spread + entry fee + exit fee, the exit fee taken at the entry
    price because the exit price is unknown when deciding.
    """
    rows = preds[preds["move"].notna()]
    up = rows[model] > 0
    price = np.where(up, rows["ask"], 1 - rows["bid"])
    # kalshi_taker_fee's formula per contract, without its per-order cent rounding
    cost = rows["ask"] - rows["bid"] + 2 * 0.07 * price * (1 - price)
    take = rows[model].abs() > cost
    net = np.where(up, 1, -1) * rows["move"] - cost
    return pd.DataFrame({"game_id": rows["game_id"], "net": net, "one": 1.0})[take]


def report(settings: Settings, progress: Callable[[str], None] = print) -> str:
    """The pre-registered evaluation as Markdown."""
    events, rejected = load_games(settings, progress)
    snaps = pd.concat([game_snapshots(g) for g in events], ignore_index=True)
    snaps["home_won"] = snaps["home_won"].astype(float)
    index = game_index(settings)
    games = index[index["game_id"].isin(snaps["game_id"])]
    progress(f"{len(snaps)} snapshots from {len(games)} games")

    lines = [
        f"Games: {len(events)} loaded, {len(rejected)} rejected by the loader, "
        f"{len(games)} with snapshots; {len(snaps):,} snapshots.",
        "",
        "| Block | Train games | Test games | Test dates | Test snapshots |",
        "|---|---|---|---|---|",
    ]
    parts = []
    for i, (train_ids, test_ids) in enumerate(walk_forward(games), 1):
        test = snaps[snaps["game_id"].isin(test_ids)]
        parts.append(fit_predict(snaps[snaps["game_id"].isin(train_ids)], test))
        days = games.loc[games["game_id"].isin(test_ids), "day"]
        lines.append(
            f"| {i} | {len(train_ids)} | {len(test_ids)} | {days.min()} to {days.max()} "
            f"| {len(test):,} |"
        )
        progress(f"block {i} done")
    preds = pd.concat(parts, ignore_index=True)
    test_games = pd.Index(preds["game_id"].unique())
    y = preds["home_won"]
    preds["one"] = 1.0
    for name in ("mid", "A0", "A1", "A2"):
        preds[f"ll_{name}"] = log_loss(preds[name], y)

    lines += [
        "",
        "**Target A: home wins** (pooled over test snapshots; lower is better)",
        "",
        "| Forecast | Log-loss | Brier |",
        "|---|---|---|",
    ]
    for name, label in (
        ("mid", "raw mid (no model)"),
        ("A0", "A0 logistic on logit(mid)"),
        ("A1", "A1 logistic, all features"),
        ("A2", "A2 gradient boosting, all features"),
    ):
        brier = float(((preds[name] - y) ** 2).mean())
        lines.append(f"| {label} | {preds[f'll_{name}'].mean():.5f} | {brier:.5f} |")

    def delta(frame: pd.DataFrame, model: str, base: str, games_: pd.Index) -> str:
        frame = frame.assign(d=frame[f"ll_{model}"] - frame[f"ll_{base}"])
        lo, hi = game_ci(frame, "d", "one", games_)
        return f"{frame['d'].mean():+.5f} | [{lo:+.5f}, {hi:+.5f}]"

    lines += [
        "",
        "Δlog-loss = model minus baseline (negative: the model beats the baseline); "
        "95% CI resamples games.",
        "",
        "| Comparison | Δlog-loss | 95% CI |",
        "|---|---|---|",
    ]
    for model in ("A1", "A2"):
        for base in ("A0", "mid"):
            lines.append(f"| {model} vs {base} | {delta(preds, model, base, test_games)} |")

    lines += ["", "**Splits, A1 vs A0**", "", "| Split | Snapshots | Δlog-loss | 95% CI |"]
    lines.append("|---|---|---|---|")
    for label, mask in (
        ("first half (periods 1-2)", preds["period"] <= 2),
        ("second half + OT", preds["period"] >= 3),
        ("|margin| ≤ 9", preds["margin"].abs() <= 9),
        ("|margin| > 9", preds["margin"].abs() > 9),
    ):
        part = preds[mask]
        part_games = pd.Index(part["game_id"].unique())
        lines.append(f"| {label} | {len(part):,} | {delta(part, 'A1', 'A0', part_games)} |")

    has_b = preds[preds["move"].notna()].copy()
    has_b["sst"] = has_b["move"] ** 2
    lines += [
        "",
        "**Target B: home mid 3 minutes later minus now** "
        "(R² out of sample, vs predicting no change)",
        "",
        f"Snapshots with a target: {len(has_b):,}.",
        "",
        "| Model | R² | 95% CI |",
        "|---|---|---|",
    ]
    for model, label in (("B1", "B1 ridge"), ("B2", "B2 gradient boosting")):
        has_b["sse"] = (has_b["move"] - has_b[model]) ** 2
        lo, hi = game_ci(has_b, "sse", "sst", test_games)
        r2 = 1 - has_b["sse"].sum() / has_b["sst"].sum()
        lines.append(f"| {label} | {r2:+.4f} | [{1 - hi:+.4f}, {1 - lo:+.4f}] |")

    lines += [
        "",
        "**Economics of B**: trade when |predicted move| > spread + taker fee on both legs; "
        "net = realized signed move minus that cost, per contract.",
        "",
        "| Model | Trades | Games | Mean net per contract | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for model in ("B1", "B2"):
        trades = taker_trades(preds, model)
        if trades.empty:
            lines.append(f"| {model} | 0 | 0 | - | - |")
            continue
        lo, hi = game_ci(trades, "net", "one", test_games)
        lines.append(
            f"| {model} | {len(trades):,} | {trades['game_id'].nunique()} "
            f"| {100 * trades['net'].mean():+.2f}¢ | [{100 * lo:+.2f}¢, {100 * hi:+.2f}¢] |"
        )
    return "\n".join(lines)
