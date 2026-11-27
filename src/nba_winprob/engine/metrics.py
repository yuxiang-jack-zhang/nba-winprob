"""Backtest summaries with a bootstrap interval over games."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summary(
    trades: pd.DataFrame, games: pd.Index, n_boot: int = 4000, seed: int = 0
) -> dict[str, float]:
    """Totals over the round trips in ``trades`` and a 95% interval of total P&L.

    The interval resamples every game in ``games``, those without trades counting as 0,
    so it describes the strategy over the whole sample rather than only games it traded.
    """
    pnl = trades.groupby("game_id")["pnl"].sum().reindex(games, fill_value=0.0).to_numpy()
    rng = np.random.default_rng(seed)
    draws = pnl[rng.integers(0, len(pnl), size=(n_boot, len(pnl)))].sum(axis=1)
    n, total, staked = len(trades), float(trades["pnl"].sum()), float(trades["staked"].sum())
    return {
        "games": len(games),
        "trades": n,
        "total_pnl": total,
        "pnl_per_trade": total / n if n else float("nan"),
        "roi": total / staked if staked else float("nan"),
        "win_rate": float((trades["pnl"] > 0).mean()) if n else float("nan"),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def summary_by(
    trades: pd.DataFrame, games: pd.Index, column: str, n_boot: int = 4000, seed: int = 0
) -> pd.DataFrame:
    """``summary`` per value of ``column`` (e.g. a meta field such as ``timeout``)."""
    rows = {
        value: summary(group, games, n_boot, seed)
        for value, group in trades.groupby(column, dropna=False)
    }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(column)
