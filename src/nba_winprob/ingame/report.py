"""The pre-registered RunReversal v1 backtest, its variants, splits and a price-reaction study.

Everything reported here was fixed in the design doc before any in-game data was
examined. Variants are reported alongside the primary run, never instead of it.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from nba_winprob.config import Settings
from nba_winprob.engine.backtest import BacktestResult, run_backtest
from nba_winprob.engine.events import Event, Quote
from nba_winprob.engine.metrics import summary, summary_by
from nba_winprob.ingame.history import game_index, load_game
from nba_winprob.strategies.run_reversal import V1, RunReversal, RunReversalParams

PRIMARY_LATENCY_S = 30.0
HOLD = RunReversalParams(take_profit=math.inf, max_hold_s=math.inf, exit_seconds_left=-math.inf)
REACTION_MINUTES = (1, 3, 5, 10)


def load_games(
    settings: Settings, progress: Callable[[str], None] = lambda _: None
) -> tuple[list[list[Event]], dict[str, str]]:
    """Every loadable game's events, and why each other game was rejected."""
    games, rejected = [], {}
    for game_id in game_index(settings)["game_id"]:
        try:
            games.append(load_game(game_id, settings))
        except Exception as exc:  # the loader's checks raise on bad data; report, don't crash
            rejected[game_id] = f"{type(exc).__name__}: {exc}"
    progress(f"loaded {len(games)} games, rejected {len(rejected)}")
    return games, rejected


def run(games: list[list[Event]], params: RunReversalParams, latency_s: float) -> BacktestResult:
    return run_backtest(RunReversal(params), games, latency_s=latency_s)


def price_reaction(games: list[list[Event]], trades: pd.DataFrame) -> pd.DataFrame:
    """Mean move of the held side's mid, k minutes after each entry fill, with a game bootstrap.

    Needs no exit rule: it asks directly whether the price drifts back toward the
    team that was run on after the trigger.
    """
    quotes: dict[str, list[Quote]] = {
        g[0].game_id: [e for e in g if isinstance(e, Quote)] for g in games
    }
    rows = []
    cols = (trades["game_id"].tolist(), trades["team"].tolist(), trades["entry_ts"].tolist())
    for game_id, team, entry_ts in zip(*cols, strict=True):
        qs = quotes[game_id]
        stamps = [q.ts for q in qs]
        i = bisect_left(stamps, entry_ts)
        if i == len(qs):
            continue
        sign = 1 if team == "home" else -1
        mid0 = (qs[i].yes_bid + qs[i].yes_ask) / 2
        row: dict[str, float | str] = {
            "game_id": game_id,
            "spread": qs[i].yes_ask - qs[i].yes_bid,
        }
        for k in REACTION_MINUTES:
            j = bisect_left(stamps, entry_ts + 60 * k)
            if j < len(qs):
                row[f"+{k}m"] = sign * ((qs[j].yes_bid + qs[j].yes_ask) / 2 - mid0)
        rows.append(row)
    frame = pd.DataFrame(rows)
    out = []
    rng = np.random.default_rng(0)
    by_game = frame.groupby("game_id")
    sums, counts = by_game.sum(numeric_only=True), by_game.count()
    for col in [f"+{k}m" for k in REACTION_MINUTES]:
        idx = rng.integers(0, len(sums), size=(4000, len(sums)))
        boot = sums[col].to_numpy()[idx].sum(1) / counts[col].to_numpy()[idx].sum(1)
        out.append(
            {
                "horizon": col,
                "n": int(frame[col].notna().sum()),
                "mean_mid_move": float(frame[col].mean()),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
            }
        )
    result = pd.DataFrame(out)
    result.attrs["mean_spread"] = float(frame["spread"].mean())
    return result


def report(settings: Settings, progress: Callable[[str], None] = print) -> str:
    """The full v1 report as Markdown."""
    games, rejected = load_games(settings, progress)
    ids = pd.Index([g[0].game_id for g in games], name="game_id")
    season_type = (
        pd.read_parquet(settings.data_dir / "papertrade" / "games.parquet")
        .set_index("game_id")["season_type"]
        .reindex(ids)
    )
    playoff_ids = ids[season_type.isin([3, 5]).to_numpy()]

    def fmt(s: Mapping[Any, Any]) -> str:
        return (
            f"| {s['trades']:.0f} | ${s['total_pnl']:,.0f} | ${s['pnl_per_trade']:.2f} "
            f"| {s['roi']:+.1%} | {s['win_rate']:.0%} "
            f"| [${s['ci_low']:,.0f}, ${s['ci_high']:,.0f}] |"
        )

    head = (
        "| Run | Trades | P&L | P&L/trade | ROI | Win rate | 95% CI of P&L |\n"
        "|---|---|---|---|---|---|---|"
    )
    lines = [
        f"Games: {len(games)} loaded, {len(rejected)} rejected by the loader's data checks.",
        "",
        head,
    ]
    primary = run(games, V1, PRIMARY_LATENCY_S)
    progress("primary run done")
    trades = primary.trades
    lines.append(f"| **v1, latency 30 s (primary)** {fmt(summary(trades, primary.games))}")
    for latency in (0.0, 60.0, 120.0):
        r = run(games, V1, latency)
        lines.append(f"| v1, latency {latency:.0f} s {fmt(summary(r.trades, r.games))}")
    hold = run(games, HOLD, PRIMARY_LATENCY_S)
    lines.append(f"| hold to settlement, 30 s {fmt(summary(hold.trades, hold.games))}")
    progress("variants done")

    lines += ["", "Splits of the primary run (each over its own games):", ""]
    for title, column in (
        ("Timeout by the run-on team (in the run or ≤30 s after)", "exit_timeout_by_run_on_team"),
        ("Half", "half"),
    ):
        lines += [
            f"**{title}**",
            "",
            "| Value | Trades | P&L | P&L/trade | ROI | Win rate | 95% CI |",
            "|---|---|---|---|---|---|---|",
        ]
        for value, row in summary_by(trades, primary.games, column).iterrows():
            lines.append(f"| {value} {fmt(row.to_dict())}")
        lines.append("")
    is_playoff = trades["game_id"].isin(playoff_ids)
    lines += [
        "**Season type**",
        "",
        "| Value | Trades | P&L | P&L/trade | ROI | Win rate | 95% CI |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.append(
        f"| regular season {fmt(summary(trades[~is_playoff], ids.difference(playoff_ids)))}"
    )
    lines.append(f"| playoffs + play-in {fmt(summary(trades[is_playoff], playoff_ids))}")

    exits = trades["exit_kind"].fillna("settle").value_counts()
    lines += ["", "Exits: " + ", ".join(f"{k} {v}" for k, v in exits.items()), ""]

    reaction = price_reaction(games, trades)
    lines += [
        "Price reaction after entry (held side's mid, cents; positive = moved our way):",
        "",
        "| Horizon | n | Mean move | 95% CI |",
        "|---|---|---|---|",
    ]
    for h in reaction.to_dict("records"):
        lines.append(
            f"| {h['horizon']} | {h['n']} | {100 * h['mean_mid_move']:+.2f}¢ "
            f"| [{100 * h['ci_low']:+.2f}, {100 * h['ci_high']:+.2f}] |"
        )
    lines.append(
        f"\nMean bid-ask spread at entry: {100 * reaction.attrs['mean_spread']:.2f}¢ "
        "(a round trip pays it once, plus fees)."
    )
    return "\n".join(lines)
