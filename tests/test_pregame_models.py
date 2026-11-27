"""The models must reproduce the notebooks' published numbers, and the corrected
features must beat the original ones.

Tolerances are stated per metric rather than global. Logistic regression and
XGBoost are deterministic given the split, so they are held tight; the MLP's
published run was unseeded, so it is held to the project's stated tolerance of
0.005 log-loss and 0.01 accuracy.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nba_winprob.data.pregame_features import build_pregame_master
from nba_winprob.models import pregame as P

# Test-season (2023) metrics as published by the notebooks.
NOTEBOOK_TEST_METRICS = {
    "logreg": {"log_loss": 0.6251, "accuracy": 0.6381, "roc_auc": 0.6989, "brier": 0.2180},
    "mlp": {"log_loss": 0.62447, "accuracy": 0.64466, "roc_auc": 0.70339, "brier": 0.21745},
    "xgboost": {"log_loss": 0.636263, "accuracy": 0.639772, "roc_auc": 0.696014, "brier": 0.222609},
}

NOTEBOOK_GROUPED_SIZES = {"train": 4506, "val": 2429, "test": 1227}
NOTEBOOK_CHRONO_SIZES = {"train": 5705, "val": 1230, "test": 1227}


@pytest.fixture(scope="module")
def master(base_tables: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    games, team_games = base_tables
    _, built = build_pregame_master(games, team_games, variant="notebook")
    return built


@pytest.fixture(scope="module")
def corrected_master(base_tables: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    games, team_games = base_tables
    _, built = build_pregame_master(games, team_games, variant="corrected")
    return built


def test_feature_counts_match_the_notebooks(master: pd.DataFrame) -> None:
    assert len(P.mlp_feature_columns(master)) == 76
    # 43, not 38: five columns are listed twice. See xgb_feature_columns.
    assert len(P.xgb_feature_columns(master)) == 43


def test_split_sizes_match_the_notebooks(master: pd.DataFrame) -> None:
    grouped = P.prepare_grouped_split(master, P.mlp_feature_columns(master))
    chrono = P.prepare_chronological_split(master, P.xgb_feature_columns(master))
    assert grouped.sizes == NOTEBOOK_GROUPED_SIZES
    assert chrono.sizes == NOTEBOOK_CHRONO_SIZES
    assert grouped.test_season == chrono.test_season == 2023


def test_no_season_appears_in_both_train_and_val(master: pd.DataFrame) -> None:
    """The grouped split must keep whole seasons together.

    If a season were split across train and val, validation log-loss would
    measure memorisation of that season rather than generalisation to a new one,
    and model selection would be made on the wrong signal.
    """
    cols = P.mlp_feature_columns(master)
    seasons = master[P.GROUP_COL].to_numpy()
    test_mask = seasons == 2023
    pool_seasons = seasons[~test_mask]

    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=P.NOTEBOOK_SPLIT_SEED)
    train_idx, val_idx = next(splitter.split(master.loc[~test_mask, cols], groups=pool_seasons))
    assert not set(pool_seasons[train_idx]) & set(pool_seasons[val_idx])
    assert 2023 not in set(pool_seasons)


@pytest.mark.parametrize("model_name", ["logreg", "xgboost"])
def test_deterministic_models_reproduce_notebook_metrics(
    master: pd.DataFrame, model_name: str
) -> None:
    if model_name == "logreg":
        data = P.prepare_grouped_split(master, P.mlp_feature_columns(master))
        _, metrics = P.train_logreg(data)
        tolerance = 1e-4  # closed-form fit, so this should be near-exact
    else:
        data = P.prepare_chronological_split(master, P.xgb_feature_columns(master))
        _, metrics = P.train_xgboost(data)
        # xgboost 3.x bins and breaks ties differently from the 2.x the notebook used.
        tolerance = 5e-3

    for metric, expected in NOTEBOOK_TEST_METRICS[model_name].items():
        assert metrics["test"][metric] == pytest.approx(expected, abs=tolerance), metric


def test_mlp_reproduces_notebook_metrics_within_tolerance(master: pd.DataFrame) -> None:
    data = P.prepare_grouped_split(master, P.mlp_feature_columns(master))
    _, _, history, metrics = P.train_mlp(data, seed=0)

    assert history.stopped_early, "40 epochs without early stopping suggests the patience broke"
    expected = NOTEBOOK_TEST_METRICS["mlp"]
    assert metrics["test"]["log_loss"] == pytest.approx(expected["log_loss"], abs=5e-3)
    assert metrics["test"]["accuracy"] == pytest.approx(expected["accuracy"], abs=1e-2)
    assert metrics["test"]["roc_auc"] == pytest.approx(expected["roc_auc"], abs=1e-2)


def test_mlp_is_reproducible_across_runs(master: pd.DataFrame) -> None:
    """Same seed, same numbers. The notebooks seeded nothing; a served model must."""
    data = P.prepare_grouped_split(master, P.mlp_feature_columns(master))
    _, _, _, first = P.train_mlp(data, seed=0)
    _, _, _, second = P.train_mlp(data, seed=0)
    assert first["test"]["log_loss"] == pytest.approx(second["test"]["log_loss"], abs=1e-12)


def test_corrected_features_beat_the_originals(
    master: pd.DataFrame, corrected_master: pd.DataFrame
) -> None:
    """Fixing the rest and home/away-split bugs must not cost accuracy.

    Asserted on logistic regression because it is deterministic, so a regression
    here is a real change rather than training noise.
    """
    original = P.train_logreg(P.prepare_grouped_split(master, P.mlp_feature_columns(master)))[1]
    fixed = P.train_logreg(
        P.prepare_grouped_split(corrected_master, P.mlp_feature_columns(corrected_master))
    )[1]
    assert fixed["test"]["log_loss"] < original["test"]["log_loss"]
    assert fixed["test"]["accuracy"] > original["test"]["accuracy"]
    assert fixed["test"]["roc_auc"] > original["test"]["roc_auc"]
