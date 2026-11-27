"""The port must reproduce the notebook's output exactly, not approximately."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_winprob.data.pregame_features import (
    build_pregame_master,
    compute_elo,
    pregame_feature_columns,
)

EXPECTED_GAMES = 8162
EXPECTED_COLUMNS = 79


def test_feature_columns_are_only_lagged(base_tables: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    _, team_games = base_tables
    cols = pregame_feature_columns(team_games)
    assert len(cols) == 18
    # A column that is not a _roll5/_season aggregate is a current-game stat and
    # must never reach the feature set.
    assert all(c.endswith(("_roll5", "_season")) for c in cols)
    for current_game_stat in ("num_shots", "fg_pct", "total_made", "homeWin"):
        assert current_game_stat not in cols


def test_reproduces_baseline_pregame_master(
    base_tables: tuple[pd.DataFrame, pd.DataFrame], baseline_master: pd.DataFrame
) -> None:
    games, team_games = base_tables
    _, master = build_pregame_master(games, team_games)

    assert list(master.columns) == list(baseline_master.columns)
    assert master.shape == (EXPECTED_GAMES, EXPECTED_COLUMNS)

    got = master.sort_values("gameId").reset_index(drop=True)
    want = baseline_master.sort_values("gameId").reset_index(drop=True)
    assert got["gameId"].equals(want["gameId"])

    for col in want.columns:
        np.testing.assert_allclose(
            got[col].to_numpy(dtype=float),
            want[col].to_numpy(dtype=float),
            rtol=0,
            atol=1e-9,
            err_msg=f"column {col} differs from the baseline",
        )


def test_elo_resets_each_season(base_tables: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    games, _ = base_tables
    elo = compute_elo(games)

    # Every season's opening game starts both teams at the initial rating, which
    # is what "per-season Elo" means and what makes elo_diff a form signal.
    first = elo.sort_values(["season", "gameDate", "gameId"]).groupby("season").first()
    assert (first["elo_home_before"] == 1500.0).all()
    assert (first["elo_away_before"] == 1500.0).all()

    # Elo is zero-sum within a game, so the pre-game ratings average to the
    # initial value across a whole season.
    for _, group in elo.groupby("season"):
        pooled = pd.concat([group["elo_home_before"], group["elo_away_before"]])
        assert abs(pooled.mean() - 1500.0) < 25.0
