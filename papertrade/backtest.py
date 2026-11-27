"""Paper-trade settled Kalshi NBA game markets: do injury and fatigue features help?

    python papertrade/backtest.py

Two model families, each fitted four ways (base, +fatigue, +injury, +both):

* ``lean`` -- logistic regression on carry-over Elo, win rates and recent margin.
* ``xgb``  -- the project's XGBoost recipe (``xgb_feature_columns`` + ``xgb_params``
  on ``build_pregame_master(variant="corrected")``), retrained on ESPN-sourced
  inputs because the Kaggle data it was built on stops in 2024. Its own rest and
  back-to-back columns are removed from "base" so the fatigue block is the only
  place rest information enters.

Walk-forward: every game is predicted by a model fitted only on earlier games.
Trades are priced at the last one-minute Kalshi quote before tip, crossing the
spread and paying the taker fee. Nothing here places an order.
"""

from __future__ import annotations

import itertools
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nba_winprob.data.pregame_features import build_pregame_master
from nba_winprob.models.metrics import brier_score, log_loss_manual
from nba_winprob.models.pregame import xgb_feature_columns, xgb_params

ROOT = Path(__file__).resolve().parents[1] / "data" / "papertrade"

STAKE = 100.0  # dollars per paper trade
MIN_EDGE = 0.03  # model probability must beat cost + fee by this much
ROSTER_WINDOW = 10  # team games used to estimate who normally plays
REST_CAP = 10  # days; also what a season opener counts as
# Refit points: each model predicts only games after its own cutoff.
CUTOFFS = ["2025-04-15", "2025-10-01", "2026-10-01"]

SHOT_STATS = [
    "num_shots", "avg_shot_distance", "total_made", "fg_pct", "avg_loc_x",
    "avg_loc_y", "shots_in_paint", "shots_midrange", "shots_long",
]  # fmt: skip
LEAN = ["elo_carry_diff", "season_winrate_diff", "winrate_diff_5", "margin10_diff"]
XGB_REST = {
    "rest_diff", "b2b_diff", "days_since_last_home", "days_since_last_away",
    "is_b2b_home", "is_b2b_away",
}  # fmt: skip
FATIGUE = [
    "f_rest_home", "f_rest_away", "f_rest_diff", "f_b2b_home", "f_b2b_away",
    "f_g7_diff", "f_g14_home", "f_g14_away", "f_g14_diff",
]  # fmt: skip
INJURY = ["inj_home", "inj_away", "inj_diff"]
BLOCKS = {"base": [], "+fatigue": FATIGUE, "+injury": INJURY, "+both": FATIGUE + INJURY}


def kalshi_fee(price: float | np.ndarray) -> float | np.ndarray:
    # shortcut: per-contract fee without Kalshi's round-up-to-a-cent per order, fine at $100 stakes
    return 0.07 * price * (1 - price)


def missing_minutes(team_games: list[tuple[str, dict[str, int]]]) -> dict[str, float]:
    """Share of a team's usual minutes held by players who sat, per game.

    ``team_games`` is one team's season in order: ``(game_id, {player: minutes})``.
    "Usual" is the average over the previous ``ROSTER_WINDOW`` games only, so the
    game being described contributes nothing but who was available for it.
    """
    history: deque[dict[str, int]] = deque(maxlen=ROSTER_WINDOW)
    out = {}
    for game_id, minutes in team_games:
        if not minutes:  # no box score (ESPN 404s ~1% of games): unknown, not "everyone sat"
            out[game_id] = 0.0
            continue
        if history:
            usual: dict[str, float] = defaultdict(float)
            for past in history:
                for player, played in past.items():
                    usual[player] += played / len(history)
            sat = sum(avg for player, avg in usual.items() if minutes.get(player, 0) == 0)
            out[game_id] = sat / 240
        else:
            out[game_id] = 0.0
        history.append(minutes)
    return out


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Game-level frame and the long (one row per team per game) frame."""
    games = pd.read_parquet(ROOT / "games.parquet").drop_duplicates("game_id")
    games["tip"] = pd.to_datetime(games["tip_utc"], utc=True)
    games["gameDate"] = pd.to_datetime(games["tip"].dt.tz_convert("America/New_York").dt.date)
    games["homeWin"] = (games["home_pts"] > games["away_pts"]).astype(float)
    games = games.sort_values("tip").reset_index(drop=True)

    sides = []
    for side, sign in (("home", 1), ("away", -1)):
        sides.append(
            pd.DataFrame(
                {
                    "game_id": games["game_id"],
                    "team_id": games[f"{side}_id"],
                    "season": games["season"],
                    "gameDate": games["gameDate"],
                    "tip": games["tip"],
                    "is_home": sign == 1,
                    "margin": sign * (games["home_pts"] - games["away_pts"]),
                }
            )
        )
    long = pd.concat(sides).sort_values(["team_id", "tip"]).reset_index(drop=True)
    return games, long


def add_fatigue(long: pd.DataFrame) -> pd.DataFrame:
    by_team_season = long.groupby(["team_id", "season"])
    rest = by_team_season["gameDate"].diff().dt.days
    long["f_rest"] = rest.fillna(REST_CAP).clip(upper=REST_CAP)
    long["f_b2b"] = (rest == 1).astype(float)
    for days in (7, 14):
        # closed="left" counts games strictly before this one
        counts = (
            long.set_index("gameDate")
            .groupby("team_id")["margin"]
            .rolling(f"{days}D", closed="left")
            .count()
        )
        long[f"f_g{days}"] = counts.fillna(0).to_numpy()
    long["margin10"] = (
        long.groupby("team_id")["margin"]
        .transform(lambda s: s.shift().rolling(10, min_periods=1).mean())
        .fillna(0.0)
    )
    return long


def add_injury(long: pd.DataFrame) -> pd.DataFrame:
    players = pd.read_parquet(ROOT / "players.parquet")
    played = {
        key: dict(zip(group["player_id"], group["minutes"], strict=True))
        for key, group in players.groupby(["game_id", "team_id"])
    }
    result: dict[tuple[str, str], float] = {}
    for (team, _season), group in long.groupby(["team_id", "season"]):
        ordered = [(gid, played.get((gid, team), {})) for gid in group["game_id"]]
        for gid, share in missing_minutes(ordered).items():
            result[(gid, team)] = share
    long["inj"] = [result[(g, t)] for g, t in zip(long["game_id"], long["team_id"], strict=True)]
    return long


def carry_elo(games: pd.DataFrame, k: float = 20.0, carry: float = 0.75) -> np.ndarray:
    """Pre-game Elo difference, regressed a quarter of the way to 1500 each new season."""
    rating: dict[str, float] = defaultdict(lambda: 1500.0)
    last_season: dict[str, int] = {}
    diff = np.empty(len(games))
    for i, g in enumerate(games.itertuples()):
        for team in (g.home_id, g.away_id):
            if last_season.get(team, g.season) != g.season:
                rating[team] = carry * rating[team] + (1 - carry) * 1500.0
            last_season[team] = g.season
        home, away = rating[g.home_id], rating[g.away_id]
        diff[i] = home - away
        expected = 1 / (1 + 10 ** ((away - home) / 400))
        rating[g.home_id] = home + k * (g.homeWin - expected)
        rating[g.away_id] = away - k * (g.homeWin - expected)
    return diff


def project_master(games: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """Feed ESPN data through the project's own feature pipeline."""
    shots = pd.read_parquet(ROOT / "shots.parquet")
    shots = shots[shots["n"] > 0]
    per_game = pd.DataFrame(
        {
            "game_id": shots["game_id"],
            "team_id": shots["team_id"],
            "num_shots": shots["n"],
            "avg_shot_distance": shots["sum_dist"] / shots["n"],
            "total_made": shots["made"],
            "fg_pct": shots["made"] / shots["n"],
            "avg_loc_x": shots["sum_x"] / shots["n"],
            "avg_loc_y": shots["sum_y"] / shots["n"],
            "shots_in_paint": shots["paint"],
            "shots_midrange": shots["mid"],
            "shots_long": shots["long"],
        }
    )
    tg = long.merge(per_game, on=["game_id", "team_id"], how="left")
    tg = tg.sort_values(["team_id", "tip"])
    for stat in SHOT_STATS:  # shift() first: a game never sees its own shots
        tg[f"{stat}_roll5"] = tg.groupby("team_id")[stat].transform(
            lambda s: s.shift().rolling(5, min_periods=1).mean()
        )
        tg[f"{stat}_season"] = tg.groupby(["team_id", "season"])[stat].transform(
            lambda s: s.shift().expanding(min_periods=1).mean()
        )

    ids = games[["game_id", "home_id", "away_id", "homeWin"]]
    tg = tg.merge(ids, on="game_id").rename(
        columns={
            "game_id": "gameId",
            "team_id": "teamId",
            "home_id": "homeTeamId",
            "away_id": "awayTeamId",
        }
    )
    base_games = games.rename(
        columns={"game_id": "gameId", "home_id": "homeTeamId", "away_id": "awayTeamId"}
    )[["gameId", "awayTeamId", "homeTeamId", "season", "homeWin", "gameDate"]].copy()
    for frame in (tg, base_games):
        for col in ("gameId", "teamId", "homeTeamId", "awayTeamId"):
            if col in frame:
                frame[col] = frame[col].astype("int64")
    _, master = build_pregame_master(base_games, tg, variant="corrected")
    return master


def build_dataset() -> pd.DataFrame:
    games, long = load()
    long = add_injury(add_fatigue(long))
    games["elo_carry_diff"] = carry_elo(games)

    wide = long.pivot(
        index="game_id",
        columns="is_home",
        values=["f_rest", "f_b2b", "f_g7", "f_g14", "inj", "margin10"],
    )
    wide.columns = [f"{name}_{'home' if is_home else 'away'}" for name, is_home in wide.columns]
    for name in ("f_rest", "f_g7", "f_g14", "inj", "margin10"):
        wide[f"{name}_diff"] = wide[f"{name}_home"] - wide[f"{name}_away"]
    games = games.merge(wide.reset_index(), on="game_id")

    master = project_master(games, long)
    games["gameId"] = games["game_id"].astype("int64")
    keep = [
        "gameId",
        "game_id",
        "tip",
        "season_type",
        "home",
        "away",
        "home_pts",
        "away_pts",
        "note",
        "elo_carry_diff",
        *wide.columns,
    ]
    data = master.merge(games[keep], on="gameId").sort_values("tip").reset_index(drop=True)
    return data.loc[:, ~data.columns.duplicated()]


def predict_walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    """Add one probability column per model, each from a fit on strictly earlier games."""
    # xgb_feature_columns matches every "*_diff" column, so the new blocks' own diffs
    # must be kept out of "base" explicitly or base would already contain them.
    added = ("f_", "inj_", "margin10_", "elo_carry_")
    xgb_base = [
        c
        for c in dict.fromkeys(xgb_feature_columns(data))
        if c not in XGB_REST and not c.startswith(added)
    ]
    cutoffs = [pd.Timestamp(c, tz="UTC") for c in CUTOFFS] + [pd.Timestamp.max.tz_localize("UTC")]
    preds = {
        f"{family} {block}": np.full(len(data), np.nan)
        for family in ("lean", "xgb")
        for block in BLOCKS
    }

    from xgboost import XGBClassifier

    for start, end in itertools.pairwise(cutoffs):
        train = (data["tip"] < start).to_numpy()
        test = ((data["tip"] >= start) & (data["tip"] < end)).to_numpy()
        if not test.any():
            continue
        y = data.loc[train, "homeWin"].to_numpy()
        for block, extra in BLOCKS.items():
            cols = LEAN + extra
            lean = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            lean.fit(data.loc[train, cols], y)
            preds[f"lean {block}"][test] = lean.predict_proba(data.loc[test, cols])[:, 1]

            cols = xgb_base + extra
            booster = XGBClassifier(**xgb_params(y))
            booster.fit(data.loc[train, cols].fillna(0.0), y)
            preds[f"xgb {block}"][test] = booster.predict_proba(data.loc[test, cols].fillna(0.0))[
                :, 1
            ]
    return data.assign(**preds)


def paper_trade(
    p_home: np.ndarray, bid: np.ndarray, ask: np.ndarray, home_won: np.ndarray
) -> np.ndarray:
    """P&L in dollars per game (0 where no trade): flat stake on the side with enough edge."""
    cost_home, cost_away = ask, 1 - bid  # buying NO on the home market costs 1 - bid
    edge_home = p_home - cost_home - kalshi_fee(cost_home)
    edge_away = (1 - p_home) - cost_away - kalshi_fee(cost_away)
    take_home = (edge_home >= MIN_EDGE) & (edge_home >= edge_away)
    take_away = (edge_away >= MIN_EDGE) & ~take_home
    pnl = np.zeros(len(p_home))
    for take, cost, won in ((take_home, cost_home, home_won), (take_away, cost_away, ~home_won)):
        per_contract = np.where(won, 1.0, 0.0) - cost - kalshi_fee(cost)
        pnl[take] = (STAKE / cost * per_contract)[take]
    return pnl


def bootstrap_ci(
    values: np.ndarray, stat=np.sum, n: int = 4000, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    draws = stat(values[idx], axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def report(frame: pd.DataFrame, title: str) -> None:
    y = frame["homeWin"].to_numpy()
    won = y == 1
    models = [c for c in frame.columns if c.startswith(("lean ", "xgb "))]
    print(f"\n=== {title}: {len(frame)} games, home wins {y.mean():.1%} ===")
    mid = ((frame["bid"] + frame["ask"]) / 2).to_numpy()
    print(
        f"{'model':15s} {'logloss':>8s} {'brier':>7s} {'acc':>6s} | {'bets':>5s} {'P&L $':>8s} {'ROI':>7s}  95% CI of P&L"
    )
    print(
        f"{'kalshi mid':15s} {log_loss_manual(y, mid):8.4f} {brier_score(y, mid):7.4f} {((mid >= 0.5) == won).mean():6.3f} |"
    )
    for name in models:
        p = frame[name].to_numpy()
        pnl = paper_trade(p, frame["bid"].to_numpy(), frame["ask"].to_numpy(), won)
        bets = int((pnl != 0).sum())
        lo, hi = bootstrap_ci(pnl) if len(frame) >= 30 else (float("nan"), float("nan"))
        roi = pnl.sum() / (bets * STAKE) if bets else 0.0
        print(
            f"{name:15s} {log_loss_manual(y, p):8.4f} {brier_score(y, p):7.4f} {((p >= 0.5) == won).mean():6.3f} |"
            f" {bets:5d} {pnl.sum():8.0f} {roi:7.1%}  [{lo:.0f}, {hi:.0f}]"
        )
    if len(frame) < 30:
        return
    print("paired change from adding both blocks (negative log-loss = better):")
    for family in ("lean", "xgb"):
        base, both = frame[f"{family} base"].to_numpy(), frame[f"{family} +both"].to_numpy()
        per_game = -(y * np.log(both) + (1 - y) * np.log(1 - both)) + (
            y * np.log(base) + (1 - y) * np.log(1 - base)
        )
        lo, hi = bootstrap_ci(per_game, stat=np.mean)
        bid, ask = frame["bid"].to_numpy(), frame["ask"].to_numpy()
        pnl_gain = paper_trade(both, bid, ask, won) - paper_trade(base, bid, ask, won)
        plo, phi = bootstrap_ci(pnl_gain)
        print(
            f"  {family:5s} log-loss {per_game.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]   P&L {pnl_gain.sum():+.0f} [{plo:+.0f}, {phi:+.0f}]"
        )


def main() -> None:
    data = predict_walk_forward(build_dataset())
    quotes = pd.read_parquet(ROOT / "kalshi.parquet")[["game_id", "bid", "ask"]]
    test = data.merge(quotes, on="game_id").dropna(subset=["lean base"])
    test.to_csv(ROOT / "predictions.csv", index=False)

    report(test, "ALL settled Kalshi games")
    report(test[test["season_type"] == 2], "regular season")
    report(test[test["season_type"] != 2], "playoffs + play-in (2025 and 2026)")
    report(test[(test["season_type"] != 2) & (test["season"] == 2026)], "2026 playoffs")

    finals = test[test["note"].str.contains("NBA Finals", na=False) & (test["season"] == 2026)]
    print("\n=== 2026 Finals, game by game (probabilities are for the HOME team) ===")
    for g in finals.itertuples():
        won = np.array([g.homeWin == 1])
        bid, ask = np.array([g.bid]), np.array([g.ask])
        probs = {
            name: finals.loc[g.Index, name]
            for name in ("lean base", "lean +both", "xgb base", "xgb +both")
        }
        pnl = {name: paper_trade(np.array([p]), bid, ask, won)[0] for name, p in probs.items()}
        print(
            f"{g.tip:%b %d} {g.away} {g.away_pts} @ {g.home} {g.home_pts} | {g.note}\n"
            f"   kalshi {g.bid:.2f}/{g.ask:.2f} | "
            + "  ".join(f"{name} {p:.3f} (${pnl[name]:+.0f})" for name, p in probs.items())
            + f"\n   rest days home/away {g.f_rest_home:.0f}/{g.f_rest_away:.0f} | games in last 14d {g.f_g14_home:.0f}/{g.f_g14_away:.0f}"
            f" | minutes missing {g.inj_home:.1%}/{g.inj_away:.1%}"
        )


if __name__ == "__main__":
    main()
