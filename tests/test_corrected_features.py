"""The two feature bugs, pinned down.

These tests document the defects concretely -- a constant column is a constant
column, and a 7.6% back-to-back rate is wrong on its face -- and prove the
corrected variant fixes them without reintroducing leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_winprob.data.pregame_features import build_pregame_master, team_game_log

# Roughly 19% of a team's games are the second half of a back-to-back in a
# modern 82-game NBA season.
TRUE_B2B_RATE = 0.19

PLACEHOLDER_COLUMNS = ["home_away_winrate", "away_home_winrate"]


@pytest.fixture(scope="module")
def variants(
    base_tables: tuple[pd.DataFrame, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    games, team_games = base_tables
    _, notebook = build_pregame_master(games, team_games, variant="notebook")
    _, corrected = build_pregame_master(games, team_games, variant="corrected")
    return notebook, corrected


def test_both_variants_have_the_same_shape(variants: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    notebook, corrected = variants
    assert notebook.shape == corrected.shape
    assert list(notebook.columns) == list(corrected.columns)


def test_notebook_variant_has_two_constant_features(
    variants: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Two of the 76 features are the literal constant 0.5 and carry no signal."""
    notebook, _ = variants
    for col in PLACEHOLDER_COLUMNS:
        assert notebook[col].nunique() == 1
        assert notebook[col].iloc[0] == 0.5


def test_notebook_split_diffs_are_redundant_copies(
    variants: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Two more features are exact affine copies of other features.

    Each row has one real rate and one 0.5 placeholder, and which one is real
    depends on the side the team played. On the home team's row ``home_winrate``
    is real, giving ``home_home_winrate - 0.5``; on the away team's row it is
    ``away_winrate`` that is real, so the difference comes out negated:
    ``0.5 - away_away_winrate``. Either way it is an affine function of a column
    already in the feature set, so it adds nothing a linear model could not see.
    """
    notebook, _ = variants
    np.testing.assert_allclose(
        notebook["home_home_minus_away_wr"], notebook["home_home_winrate"] - 0.5, atol=1e-12
    )
    np.testing.assert_allclose(
        notebook["away_home_minus_away_wr"], 0.5 - notebook["away_away_winrate"], atol=1e-12
    )


def test_corrected_variant_removes_the_constants(
    variants: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    _, corrected = variants
    for col in PLACEHOLDER_COLUMNS:
        assert corrected[col].nunique() > 100
        assert 0.0 <= corrected[col].min() <= corrected[col].max() <= 1.0


def test_corrected_back_to_back_rate_is_plausible(
    variants: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The notebook's b2b flags fire far too rarely; the corrected ones don't."""
    notebook, corrected = variants
    for side in ("home", "away"):
        before = notebook[f"is_b2b_{side}"].mean()
        after = corrected[f"is_b2b_{side}"].mean()
        assert after > before
        assert abs(after - TRUE_B2B_RATE) < abs(before - TRUE_B2B_RATE)
    # The notebook's home rate was less than half the true rate.
    assert notebook["is_b2b_home"].mean() < 0.10


def test_corrected_rest_counts_games_on_both_sides(
    base_tables: tuple[pd.DataFrame, pd.DataFrame],
    variants: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Independent check against the team's full schedule for sampled games."""
    games, _ = base_tables
    _, corrected = variants
    log = team_game_log(games)
    per_team_date = log.groupby(["teamId", "gameDate"]).size()

    rng = np.random.default_rng(0)
    checked = 0
    for game_id in rng.permutation(corrected["gameId"].to_numpy()):
        row = games.loc[games["gameId"] == game_id].iloc[0]
        for side in ("home", "away"):
            team = int(row[f"{side}TeamId"])
            if per_team_date.get((team, row["gameDate"]), 0) != 1:
                continue
            prior = log[(log["teamId"] == team) & (log["gameDate"] < row["gameDate"])]
            if prior.empty:
                continue
            expected = (row["gameDate"] - prior["gameDate"].max()).days
            actual = corrected.loc[corrected["gameId"] == game_id, f"days_since_last_{side}"]
            assert float(actual.iloc[0]) == pytest.approx(float(expected))
            checked += 1
        if checked >= 30:
            break
    assert checked >= 30


def test_corrected_variant_is_still_leak_free(
    base_tables: tuple[pd.DataFrame, pd.DataFrame],
    variants: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The fix must not buy accuracy by peeking at the target game.

    Same invariance check as ``test_no_leakage`` -- reversing a game's outcome
    must leave that game's own features untouched -- applied to the new code
    path, since that is where a new leak would hide.
    """
    games, team_games = base_tables
    _, corrected = variants
    indexed = corrected.set_index("gameId", drop=False)
    feature_cols = [c for c in corrected.columns if c != "homeWin"]

    rng = np.random.default_rng(0)
    for game_id in rng.choice(corrected["gameId"].to_numpy(), size=15, replace=False):
        flipped_games, flipped_team_games = games.copy(), team_games.copy()
        for frame in (flipped_games, flipped_team_games):
            mask = frame["gameId"] == game_id
            frame.loc[mask, "homeWin"] = 1 - frame.loc[mask, "homeWin"]
        _, rebuilt = build_pregame_master(flipped_games, flipped_team_games, variant="corrected")
        rebuilt = rebuilt.set_index("gameId", drop=False)

        np.testing.assert_allclose(
            rebuilt.loc[game_id, feature_cols].to_numpy(dtype=float),
            indexed.loc[game_id, feature_cols].to_numpy(dtype=float),
            atol=0,
            rtol=0,
            err_msg=f"game {game_id}: corrected features read the target game",
        )
