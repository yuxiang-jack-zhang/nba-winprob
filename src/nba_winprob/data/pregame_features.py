"""Pregame feature tables, ported from ``notebook/feature_eng.ipynb``.

Builds ``pregame_model`` (shot-profile rolling/season features, home/away/diff) and
then ``pregame_master`` (the same plus rest, win-rate and Elo context) from the two
base tables ``games`` and ``team_games``.

**Leak-free by construction.** Every team-level feature is produced by
``.shift()`` before any ``rolling``/``expanding`` window, so a value attached to
game *g* is computed only from that team's games strictly before *g*. The
``_roll5``/``_season`` columns in ``team_games`` were built the same way upstream.
``tests/test_no_leakage.py`` proves this rather than asserting it.

This is a faithful port: it reproduces the committed ``data/baseline/pregame_master.csv``
column-for-column. Two quirks of the original are preserved deliberately and are
documented at their call sites: :func:`add_rest_features` measures rest within a
single side, and the ``*_home_minus_away_wr`` columns are differenced against a 0.5
placeholder. Both are noted in ``RESULTS.md``; changing either would break the
reproduction, so they are left for a separate, measured change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Side = Literal["home", "away"]

#: ``"notebook"`` reproduces the class project exactly, including the two feature
#: bugs described in :func:`build_rest_table` and :func:`add_team_winrates`.
#: ``"corrected"`` fixes both. Every model can be trained either way, and
#: ``RESULTS.md`` reports both, so the cost of each bug is measured rather than
#: assumed. The default stays ``"notebook"`` so the reproduction test is the
#: thing that runs by default.
FeatureVariant = Literal["notebook", "corrected"]

GAME_ID_COLS = ["gameId", "homeTeamId", "awayTeamId"]
TEAM_GAME_ID_COLS = ["gameId", "teamId", "homeTeamId", "awayTeamId"]

# Context columns merged onto pregame_model to make pregame_master, in the
# original's order so the output column order matches the baseline exactly.
CONTEXT_COLS = [
    "gameId",
    "days_since_last_home",
    "days_since_last_away",
    "is_b2b_home",
    "is_b2b_away",
    "rest_diff",
    "b2b_diff",
    "home_winrate_5",
    "away_winrate_5",
    "winrate_diff_5",
    "home_season_winrate",
    "away_season_winrate",
    "season_winrate_diff",
    "home_home_winrate",
    "home_away_winrate",
    "away_home_winrate",
    "away_away_winrate",
    "home_home_minus_away_wr",
    "away_home_minus_away_wr",
    "home_home_vs_away_away_wr_diff",
    "elo_home_before",
    "elo_away_before",
    "elo_diff",
]

# Win rates get a neutral 0.5 prior when a team has no history yet; differences
# of unknown rates get 0.0. Order matters: the diffs are computed from unfilled
# rates, so a game can have a 0.5 rate and a 0.0 diff.
_RATE_COLS = [
    "home_winrate_5",
    "away_winrate_5",
    "home_season_winrate",
    "away_season_winrate",
    "home_home_winrate",
    "home_away_winrate",
    "away_home_winrate",
    "away_away_winrate",
]
_DIFF_COLS = [
    "winrate_diff_5",
    "season_winrate_diff",
    "home_home_minus_away_wr",
    "away_home_minus_away_wr",
    "home_home_vs_away_away_wr_diff",
]

# The win-rate slice of CONTEXT_COLS: what build_team_context contributes, as
# opposed to the rest columns (from build_rest_table) and Elo (from compute_elo).
WINRATE_CTX_COLS = [
    "home_winrate_5",
    "away_winrate_5",
    "winrate_diff_5",
    "home_season_winrate",
    "away_season_winrate",
    "season_winrate_diff",
    "home_home_winrate",
    "home_away_winrate",
    "away_home_winrate",
    "away_away_winrate",
    "home_home_minus_away_wr",
    "away_home_minus_away_wr",
    "home_home_vs_away_away_wr_diff",
]


def load_base_tables(games_path: Path, team_games_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean ``games`` and ``team_games``.

    Drops games with no label, restricts ``team_games`` to labelled games, parses
    dates, and coerces the ID columns to int64 (they arrive as floats with a few
    missing values).
    """
    games = pd.read_csv(games_path, low_memory=False)
    team_games = pd.read_csv(team_games_path, low_memory=False)

    games = games.dropna(subset=["homeWin"]).copy()
    team_games = team_games[team_games["gameId"].isin(games["gameId"])].copy()

    games["gameDate"] = pd.to_datetime(games["gameDate"])
    team_games["gameDate"] = pd.to_datetime(team_games["gameDate"])

    games = _coerce_ids(games, GAME_ID_COLS)
    team_games = _coerce_ids(team_games, TEAM_GAME_ID_COLS)
    return games, team_games


def _coerce_ids(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    for col in id_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=id_cols)
    df[id_cols] = df[id_cols].astype("int64")
    return df


def pregame_feature_columns(team_games: pd.DataFrame) -> list[str]:
    """The lagged shot-profile columns: the ``_roll5`` and ``_season`` pairs (18 of them).

    Only lagged columns qualify, which is what keeps the current game's own box
    score out of its features.
    """
    return [c for c in team_games.columns if c.endswith(("_roll5", "_season"))]


def build_pregame_model(
    games: pd.DataFrame, team_games: pd.DataFrame, feat_cols: list[str] | None = None
) -> pd.DataFrame:
    """Pivot ``team_games`` to one row per game with ``_home``, ``_away`` and ``_diff`` features.

    Games missing any difference feature -- a team's first game, with no history --
    are dropped, which is what takes 9,407 games down to 8,162.
    """
    if feat_cols is None:
        feat_cols = pregame_feature_columns(team_games)

    home_pre = team_games.loc[
        team_games["teamId"] == team_games["homeTeamId"], ["gameId", *feat_cols]
    ].copy()
    home_pre = home_pre.add_suffix("_home").rename(columns={"gameId_home": "gameId"})

    away_pre = team_games.loc[
        team_games["teamId"] == team_games["awayTeamId"], ["gameId", *feat_cols]
    ].copy()
    away_pre = away_pre.add_suffix("_away").rename(columns={"gameId_away": "gameId"})

    pregame = (
        games[["gameId", "homeWin", "season"]]
        .merge(home_pre, on="gameId", how="inner")
        .merge(away_pre, on="gameId", how="inner")
    )

    for col in feat_cols:
        pregame[f"{col}_diff"] = pregame[f"{col}_home"] - pregame[f"{col}_away"]

    diff_cols = [c for c in pregame.columns if c.endswith("_diff")]
    return pregame.dropna(subset=diff_cols).copy()


def add_rest_features(games_df: pd.DataFrame, side: Side) -> pd.DataFrame:
    """Add ``days_since_last_<side>`` and ``is_b2b_<side>``.

    Quirk preserved from the original: the gap is measured against that team's
    previous game *on the same side* -- for ``side="home"``, its previous home
    game -- because only one side's rows are in scope here. A true back-to-back
    flag would consider games of either kind, so these columns understate rest
    pressure. Documented in ``RESULTS.md``; changing it moves the metrics.
    """
    team_col = f"{side}TeamId"
    df = games_df[["gameId", "gameDate", team_col]].rename(columns={team_col: "teamId"})
    df = df.sort_values(["teamId", "gameDate"], kind="mergesort")

    df[f"days_since_last_{side}"] = df.groupby("teamId")["gameDate"].diff().dt.days
    df[f"is_b2b_{side}"] = (df[f"days_since_last_{side}"] == 1).astype(float)

    return games_df.merge(
        df[["gameId", "teamId", f"days_since_last_{side}", f"is_b2b_{side}"]],
        left_on=["gameId", team_col],
        right_on=["gameId", "teamId"],
        how="left",
    ).drop(columns=["teamId"])


def team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team) from both sides, ordered by team and date.

    A team's schedule is the union of its home and away games, so this is the
    frame any true rest calculation has to work from.
    """
    home = games[["gameId", "gameDate", "homeTeamId"]].rename(columns={"homeTeamId": "teamId"})
    away = games[["gameId", "gameDate", "awayTeamId"]].rename(columns={"awayTeamId": "teamId"})
    log = pd.concat([home, away], ignore_index=True)
    return log.sort_values(["teamId", "gameDate"], kind="mergesort").reset_index(drop=True)


def build_rest_table(games: pd.DataFrame, variant: FeatureVariant = "notebook") -> pd.DataFrame:
    """Rest and back-to-back context for both sides, plus their differences.

    **Bug in the ``"notebook"`` variant.** :func:`add_rest_features` sees only one
    side's rows, so ``days_since_last_home`` is the gap since that team's previous
    *home* game and ``is_b2b_home`` means "played at home yesterday". Real
    back-to-backs are mostly home-then-away or away-then-home, so the flags fire
    far too rarely and asymmetrically: 7.6% of games for the home team and 13.0%
    for the away team, against a true per-team rate of roughly 19%.

    The ``"corrected"`` variant measures the gap against the team's previous game
    on either side.
    """
    if variant == "notebook":
        games_rest = add_rest_features(games.copy(), "home")
        games_rest = add_rest_features(games_rest, "away")
    else:
        games_rest = _rest_from_full_schedule(games)

    # b2b_diff is taken before the fill; is_b2b is already 0.0 wherever the gap
    # was missing, so the fill would not change it.
    games_rest["b2b_diff"] = games_rest["is_b2b_home"] - games_rest["is_b2b_away"]
    for col in ("days_since_last_home", "days_since_last_away"):
        games_rest[col] = games_rest[col].fillna(0.0)
    games_rest["rest_diff"] = (
        games_rest["days_since_last_home"] - games_rest["days_since_last_away"]
    )
    return games_rest


def _rest_from_full_schedule(games: pd.DataFrame) -> pd.DataFrame:
    """Rest measured against each team's previous game on either side."""
    log = team_game_log(games)
    log["days_since_last"] = log.groupby("teamId")["gameDate"].diff().dt.days
    log["is_b2b"] = (log["days_since_last"] == 1).astype(float)

    out = games.copy()
    for side in ("home", "away"):
        merged = out[["gameId", f"{side}TeamId"]].merge(
            log[["gameId", "teamId", "days_since_last", "is_b2b"]],
            left_on=["gameId", f"{side}TeamId"],
            right_on=["gameId", "teamId"],
            how="left",
        )
        out[f"days_since_last_{side}"] = merged["days_since_last"].to_numpy()
        out[f"is_b2b_{side}"] = merged["is_b2b"].to_numpy()
    return out


def add_team_winrates(
    team_games: pd.DataFrame, variant: FeatureVariant = "notebook"
) -> pd.DataFrame:
    """Add rolling, season-to-date and home/away-split win rates to ``team_games``.

    Every window is preceded by ``.shift()``, so no row sees its own result. The
    rolling 5-game rate deliberately spans season boundaries; the season-to-date
    and split rates reset each season.

    **Bug in the ``"notebook"`` variant.** The home-only rate is computed on home
    rows only and the away-only rate on away rows only, then both are filled with
    0.5. Every row therefore has one real rate and one 0.5 placeholder. Carried
    into ``pregame_master`` that makes ``home_away_winrate`` and
    ``away_home_winrate`` **constant 0.5** -- two of the 76 features carry no
    information at all -- and makes ``home_home_minus_away_wr`` and
    ``away_home_minus_away_wr`` exact affine copies of ``home_home_winrate`` and
    ``away_away_winrate``. So four of the 76 features are constant or redundant.

    The ``"corrected"`` variant gives every row both rates, each computed from
    that team's preceding games in that role within the season, so
    ``home_minus_away_wr`` becomes the home-court split it was meant to be.
    """
    tg = team_games.sort_values(["teamId", "gameDate"], kind="mergesort").copy()

    # 1 if this row's team won, whichever side it played.
    tg["team_win"] = np.where(tg["teamId"] == tg["homeTeamId"], tg["homeWin"], 1 - tg["homeWin"])

    tg["winrate_5"] = tg.groupby("teamId")["team_win"].transform(
        lambda s: s.shift().rolling(5, min_periods=1).mean()
    )

    tg["season_winrate"] = tg.groupby(["teamId", "season"])["team_win"].transform(
        lambda s: s.shift().expanding(min_periods=1).mean()
    )
    tg["season_winrate"] = tg["season_winrate"].fillna(0.5)

    tg["is_home"] = (tg["teamId"] == tg["homeTeamId"]).astype(int)
    tg["is_away"] = (tg["teamId"] == tg["awayTeamId"]).astype(int)
    tg = tg.sort_values(["teamId", "season", "gameDate"], kind="mergesort")

    for mask_col, out_col in (("is_home", "home_winrate"), ("is_away", "away_winrate")):
        mask = tg[mask_col] == 1
        if variant == "notebook":
            tg.loc[mask, out_col] = (
                tg[mask]
                .groupby(["teamId", "season"])["team_win"]
                .transform(lambda s: s.shift().expanding(min_periods=1).mean())
            )
            tg[out_col] = tg[out_col].fillna(0.5)
        else:
            tg[out_col] = _prior_role_winrate(tg, mask)

    tg["home_minus_away_wr"] = tg["home_winrate"] - tg["away_winrate"]
    return tg


def _prior_role_winrate(tg: pd.DataFrame, role_mask: pd.Series) -> NDArray[np.float64]:
    """Season-to-date win rate over this team's preceding games in one role, on every row.

    Defined for rows of either role, which is the point: an away row still needs
    to know the team's home record. Computed from cumulative sums minus the
    current row's own contribution, so the current game is always excluded --
    the same leak-free guarantee ``.shift()`` gives, but available everywhere.

    Expects ``tg`` sorted by ``(teamId, season, gameDate)``.
    """
    is_role = role_mask.to_numpy(dtype=float)
    frame = pd.DataFrame(
        {
            "teamId": tg["teamId"].to_numpy(),
            "season": tg["season"].to_numpy(),
            "is_role": is_role,
            "role_win": tg["team_win"].to_numpy(dtype=float) * is_role,
        }
    )
    grouped = frame.groupby(["teamId", "season"], sort=False)
    prior_games = grouped["is_role"].cumsum().to_numpy() - is_role
    prior_wins = grouped["role_win"].cumsum().to_numpy() - frame["role_win"].to_numpy()

    # No prior games in this role this season means the neutral 0.5 prior.
    return np.where(prior_games > 0, prior_wins / np.maximum(prior_games, 1.0), 0.5)


def compute_elo(games: pd.DataFrame, k: float = 20.0, init: float = 1500.0) -> pd.DataFrame:
    """Per-season Elo, returning each game's pre-game ratings.

    Ratings reset to ``init`` at the start of every season, so Elo measures form
    within a season rather than franchise strength across decades. Games are
    processed in ``(season, gameDate, gameId)`` order, which is a total order, so
    the result is deterministic.
    """
    games_sorted = games.sort_values(["season", "gameDate", "gameId"]).reset_index(drop=True)

    seasons = games_sorted["season"].to_numpy()
    home_ids = games_sorted["homeTeamId"].to_numpy(dtype=np.int64)
    away_ids = games_sorted["awayTeamId"].to_numpy(dtype=np.int64)
    home_wins = games_sorted["homeWin"].to_numpy(dtype=np.int64)

    elo_home_before = np.empty(len(games_sorted), dtype=np.float64)
    elo_away_before = np.empty(len(games_sorted), dtype=np.float64)

    ratings: dict[int, float] = {}
    current_season: object = None

    for i in range(len(games_sorted)):
        if seasons[i] != current_season:
            current_season = seasons[i]
            ratings = {}

        home, away = int(home_ids[i]), int(away_ids[i])
        elo_home = ratings.get(home, init)
        elo_away = ratings.get(away, init)
        elo_home_before[i] = elo_home
        elo_away_before[i] = elo_away

        expected_home = 1.0 / (1.0 + 10.0 ** ((elo_away - elo_home) / 400.0))
        score_home = float(home_wins[i])
        ratings[home] = elo_home + k * (score_home - expected_home)
        ratings[away] = elo_away + k * ((1.0 - score_home) - (1.0 - expected_home))

    games_sorted["elo_home_before"] = elo_home_before
    games_sorted["elo_away_before"] = elo_away_before
    games_sorted["elo_diff"] = elo_home_before - elo_away_before

    return games.merge(
        games_sorted[["gameId", "elo_home_before", "elo_away_before", "elo_diff"]],
        on="gameId",
        how="left",
    )


def build_team_context(games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """Join each game's two teams' win-rate features onto one game-level row."""
    team_feats = team_games[
        [
            "gameId",
            "teamId",
            "winrate_5",
            "season_winrate",
            "home_winrate",
            "away_winrate",
            "home_minus_away_wr",
        ]
    ].copy()

    renames: dict[Side, dict[str, str]] = {
        "home": {
            "winrate_5": "home_winrate_5",
            "season_winrate": "home_season_winrate",
            "home_winrate": "home_home_winrate",
            "away_winrate": "home_away_winrate",
            "home_minus_away_wr": "home_home_minus_away_wr",
        },
        "away": {
            "winrate_5": "away_winrate_5",
            "season_winrate": "away_season_winrate",
            "home_winrate": "away_home_winrate",
            "away_winrate": "away_away_winrate",
            "home_minus_away_wr": "away_home_minus_away_wr",
        },
    }

    ctx = games[["gameId", "homeWin", "season", "gameDate", "homeTeamId", "awayTeamId"]].copy()
    for side in ("home", "away"):
        mapping = renames[side]
        side_feats = (
            games[["gameId", f"{side}TeamId"]]
            .merge(
                team_feats,
                left_on=["gameId", f"{side}TeamId"],
                right_on=["gameId", "teamId"],
                how="left",
            )
            .drop(columns=["teamId"])
            .rename(columns=mapping)
        )
        ctx = ctx.merge(side_feats[["gameId", *mapping.values()]], on="gameId", how="left")

    ctx["winrate_diff_5"] = ctx["home_winrate_5"] - ctx["away_winrate_5"]
    ctx["season_winrate_diff"] = ctx["home_season_winrate"] - ctx["away_season_winrate"]
    ctx["home_home_vs_away_away_wr_diff"] = ctx["home_home_winrate"] - ctx["away_away_winrate"]

    for col in _RATE_COLS:
        ctx[col] = ctx[col].fillna(0.5)
    for col in _DIFF_COLS:
        ctx[col] = ctx[col].fillna(0.0)
    return ctx


def build_pregame_master(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    k: float = 20.0,
    init: float = 1500.0,
    variant: FeatureVariant = "notebook",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ``(pregame_model, pregame_master)`` from the cleaned base tables.

    ``variant="notebook"`` reproduces the class project's output exactly;
    ``"corrected"`` fixes the rest and home/away-split bugs. Both produce the
    same 79 columns, so downstream code and the models are unchanged.
    """
    pregame_model = build_pregame_model(games, team_games)

    team_games = add_team_winrates(team_games, variant=variant)
    games_rest = build_rest_table(games, variant=variant)
    games_elo = compute_elo(games, k=k, init=init)
    team_ctx = build_team_context(games, team_games)

    games_ctx = games_rest.merge(team_ctx[["gameId", *WINRATE_CTX_COLS]], on="gameId", how="left")
    games_ctx = games_ctx.merge(
        games_elo[["gameId", "elo_home_before", "elo_away_before", "elo_diff"]],
        on="gameId",
        how="left",
    )

    pregame_master = pregame_model.merge(games_ctx[CONTEXT_COLS], on="gameId", how="left")
    return pregame_model, pregame_master
