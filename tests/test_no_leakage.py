"""Proof that pregame features never see the game they describe.

An earlier milestone version of this project leaked target-game statistics into
the pregame feature set; the rebuild fixed it. These tests exist so that property
is *checked* rather than asserted in a comment, because it is the one bug in this
pipeline that inflates every metric while leaving all of them looking plausible.

Two independent angles:

* :func:`test_flipping_a_games_label_leaves_its_own_features_untouched` is the
  decisive one. The label is the only thing a pregame feature could leak, so
  flipping a single game's outcome and rebuilding must leave that game's own
  features bit-identical. A companion test confirms the flip *does* move later
  games, so this cannot pass vacuously.
* The recomputation tests rebuild two win-rate features from scratch for sampled
  games, from only the strictly-preceding games, and compare.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_winprob.data.pregame_features import build_pregame_master

SAMPLE_SIZE = 40
RNG_SEED = 0


def _flip_label(
    games: pd.DataFrame, team_games: pd.DataFrame, game_id: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copies of the base tables with one game's outcome reversed in both."""
    flipped_games = games.copy()
    flipped_team_games = team_games.copy()
    for frame in (flipped_games, flipped_team_games):
        mask = frame["gameId"] == game_id
        assert mask.any(), f"gameId {game_id} not present"
        frame.loc[mask, "homeWin"] = 1 - frame.loc[mask, "homeWin"]
    return flipped_games, flipped_team_games


def _sample_game_ids(master: pd.DataFrame, size: int = SAMPLE_SIZE) -> list[int]:
    rng = np.random.default_rng(RNG_SEED)
    ids = master["gameId"].to_numpy()
    return [int(i) for i in rng.choice(ids, size=size, replace=False)]


@pytest.fixture(scope="module")
def master(base_tables: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    games, team_games = base_tables
    _, built = build_pregame_master(games, team_games)
    return built.set_index("gameId", drop=False)


def test_flipping_a_games_label_leaves_its_own_features_untouched(
    base_tables: tuple[pd.DataFrame, pd.DataFrame], master: pd.DataFrame
) -> None:
    games, team_games = base_tables
    feature_cols = [c for c in master.columns if c != "homeWin"]

    for game_id in _sample_game_ids(master):
        flipped_games, flipped_team_games = _flip_label(games, team_games, game_id)
        _, rebuilt = build_pregame_master(flipped_games, flipped_team_games)
        rebuilt = rebuilt.set_index("gameId", drop=False)

        before = master.loc[game_id, feature_cols].to_numpy(dtype=float)
        after = rebuilt.loc[game_id, feature_cols].to_numpy(dtype=float)
        changed = [
            col
            for col, b, a in zip(feature_cols, before, after, strict=True)
            if not (np.isnan(b) and np.isnan(a)) and b != a
        ]
        assert not changed, (
            f"game {game_id}: reversing its outcome changed its own features "
            f"{changed} -- those features are reading the target game"
        )

        # Control: the flip must be visible somewhere, or this proves nothing.
        assert rebuilt.loc[game_id, "homeWin"] != master.loc[game_id, "homeWin"]


def test_flipping_a_label_does_move_later_games(
    base_tables: tuple[pd.DataFrame, pd.DataFrame], master: pd.DataFrame
) -> None:
    """Guards the test above against passing vacuously.

    If the features ignored outcomes altogether the invariance test would pass
    for the wrong reason, so confirm a flip propagates forward.
    """
    games, team_games = base_tables
    # The season opener for the earliest season: almost everything comes after it.
    earliest = games.sort_values(["season", "gameDate", "gameId"]).iloc[0]
    game_id = int(earliest["gameId"])

    flipped_games, flipped_team_games = _flip_label(games, team_games, game_id)
    _, rebuilt = build_pregame_master(flipped_games, flipped_team_games)

    before = master.reset_index(drop=True).sort_values("gameId").reset_index(drop=True)
    after = rebuilt.sort_values("gameId").reset_index(drop=True)
    moved = [
        col
        for col in ("home_season_winrate", "elo_home_before", "home_winrate_5")
        if not np.allclose(
            before[col].to_numpy(dtype=float), after[col].to_numpy(dtype=float), atol=1e-12
        )
    ]
    assert moved, "flipping the first game of the dataset changed nothing downstream"


def _team_win_series(team_games: pd.DataFrame) -> pd.Series:
    """Independent reimplementation: did this row's team win this game?"""
    return pd.Series(
        np.where(
            team_games["teamId"] == team_games["homeTeamId"],
            team_games["homeWin"],
            1 - team_games["homeWin"],
        ),
        index=team_games.index,
        dtype=float,
    )


@pytest.fixture(scope="module")
def team_history(base_tables: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    _, team_games = base_tables
    history = team_games[["gameId", "teamId", "season", "gameDate"]].copy()
    history["team_win"] = _team_win_series(team_games)
    return history


def _unambiguous_samples(
    games: pd.DataFrame, master: pd.DataFrame, history: pd.DataFrame, size: int
) -> list[tuple[int, str]]:
    """Sampled ``(gameId, side)`` pairs where the team plays once on that date.

    The pipeline orders a team's games by date, so a team playing twice on one
    date has an ambiguous ordering that an independent recomputation cannot
    mirror. NBA teams do not, but the filter keeps the test honest.
    """
    per_team_date = history.groupby(["teamId", "gameDate"]).size()
    rng = np.random.default_rng(RNG_SEED)
    candidates = master["gameId"].to_numpy()
    picked: list[tuple[int, str]] = []
    for game_id in rng.permutation(candidates):
        row = games.loc[games["gameId"] == game_id].iloc[0]
        for side in ("home", "away"):
            team = int(row[f"{side}TeamId"])
            if per_team_date.get((team, row["gameDate"]), 0) == 1:
                picked.append((int(game_id), side))
        if len(picked) >= size:
            break
    return picked[:size]


def test_season_winrate_matches_independent_recomputation(
    base_tables: tuple[pd.DataFrame, pd.DataFrame],
    master: pd.DataFrame,
    team_history: pd.DataFrame,
) -> None:
    games, _ = base_tables
    for game_id, side in _unambiguous_samples(games, master, team_history, SAMPLE_SIZE):
        row = games.loc[games["gameId"] == game_id].iloc[0]
        team = int(row[f"{side}TeamId"])

        prior = team_history[
            (team_history["teamId"] == team)
            & (team_history["season"] == row["season"])
            & (team_history["gameDate"] < row["gameDate"])
        ]
        # No history yet means the neutral 0.5 prior, not a missing value.
        expected = float(prior["team_win"].mean()) if len(prior) else 0.5
        actual = float(master.loc[game_id, f"{side}_season_winrate"])
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"game {game_id} {side}: season win rate {actual} != {expected} "
            f"recomputed from its {len(prior)} preceding games"
        )


def test_rolling_winrate_matches_independent_recomputation(
    base_tables: tuple[pd.DataFrame, pd.DataFrame],
    master: pd.DataFrame,
    team_history: pd.DataFrame,
) -> None:
    games, _ = base_tables
    for game_id, side in _unambiguous_samples(games, master, team_history, SAMPLE_SIZE):
        row = games.loc[games["gameId"] == game_id].iloc[0]
        team = int(row[f"{side}TeamId"])

        # The 5-game window deliberately spans seasons, so no season filter here.
        prior = team_history[
            (team_history["teamId"] == team) & (team_history["gameDate"] < row["gameDate"])
        ].sort_values("gameDate", kind="mergesort")
        window = prior["team_win"].tail(5)
        expected = float(window.mean()) if len(window) else 0.5
        actual = float(master.loc[game_id, f"{side}_winrate_5"])
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"game {game_id} {side}: rolling win rate {actual} != {expected} "
            f"recomputed from its last {len(window)} preceding games"
        )
