"""Pregame win-probability models: logistic baseline, MLP, XGBoost.

Ported from ``pregame_nn.ipynb`` and ``pregame_XGBoost.ipynb``.

**The three models were not trained the same way**, and that is preserved here
rather than tidied away, because tidying it would silently change the numbers the
project reports:

* The logistic baseline and the MLP use all 76 feature columns and carve a
  validation set out of the pre-test seasons with ``GroupShuffleSplit`` (whole
  seasons, ``random_state=42``).
* XGBoost uses 38 columns (the 18 differentials plus 20 named context columns)
  and a strictly chronological split: validation is the penultimate season.

So the comparison in ``RESULTS.md`` is between three recipes, not three model
classes on equal footing. ``RESULTS.md`` says so.

Both splits hold out the latest season (2023) for test and never shuffle games
across seasons, so no model is trained on games played after its test games.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nba_winprob.models.metrics import ClassificationMetrics, classification_metrics

if TYPE_CHECKING:
    from xgboost import XGBClassifier

LABEL_COL = "homeWin"
GROUP_COL = "season"
ID_COLS = ["gameId", LABEL_COL, GROUP_COL]

# The notebooks' validation split seed. It decides which seasons land in
# validation, so changing it changes the selected model, not just the noise.
NOTEBOOK_SPLIT_SEED = 42

# XGBoost's hand-picked context columns, in the notebook's order.
XGB_CONTEXT_COLS = [
    "rest_diff",
    "b2b_diff",
    "days_since_last_home",
    "days_since_last_away",
    "is_b2b_home",
    "is_b2b_away",
    "winrate_diff_5",
    "home_winrate_5",
    "away_winrate_5",
    "season_winrate_diff",
    "home_season_winrate",
    "away_season_winrate",
    "home_home_winrate",
    "home_away_winrate",
    "away_home_winrate",
    "away_away_winrate",
    "home_home_vs_away_away_wr_diff",
    "elo_diff",
    "elo_home_before",
    "elo_away_before",
]

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class PregameData:
    """Feature matrices and labels for one recipe's train/val/test split."""

    x_train: NDArray[np.float64]
    y_train: NDArray[np.float64]
    x_val: NDArray[np.float64]
    y_val: NDArray[np.float64]
    x_test: NDArray[np.float64]
    y_test: NDArray[np.float64]
    feature_cols: list[str]
    test_season: int

    def split(self, name: Split) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return {
            "train": (self.x_train, self.y_train),
            "val": (self.x_val, self.y_val),
            "test": (self.x_test, self.y_test),
        }[name]

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.y_train),
            "val": len(self.y_val),
            "test": len(self.y_test),
        }


def mlp_feature_columns(df: pd.DataFrame) -> list[str]:
    """All 76 non-identifier columns: home, away, diff and context features."""
    return [c for c in df.columns if c not in ID_COLS]


def xgb_feature_columns(df: pd.DataFrame) -> list[str]:
    """The notebook's 43 XGBoost columns: 23 ``*_diff`` plus the 20 named context ones.

    Five columns appear twice. ``rest_diff``, ``b2b_diff``, ``season_winrate_diff``,
    ``home_home_vs_away_away_wr_diff`` and ``elo_diff`` all end in ``_diff``, so they
    are picked up by the suffix match *and* listed explicitly in
    :data:`XGB_CONTEXT_COLS`, and the notebook concatenated the two lists without
    deduplicating. The duplication is preserved: it is what the published
    XGBoost metrics were produced with, and removing it changes them.
    """
    diff_cols = [c for c in df.columns if c.endswith("_diff")]
    context = [c for c in XGB_CONTEXT_COLS if c in df.columns]
    return diff_cols + context


def _latest_seasons(df: pd.DataFrame) -> list[int]:
    return [int(s) for s in sorted(pd.unique(df[GROUP_COL].dropna()))]


def prepare_grouped_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    test_season: int | None = None,
    val_fraction: float = 0.2,
    split_seed: int = NOTEBOOK_SPLIT_SEED,
) -> PregameData:
    """Hold out ``test_season``, then split the rest into whole-season train/val.

    Used by the logistic baseline and the MLP. ``GroupShuffleSplit`` grouped on
    season guarantees no season is partly in train and partly in validation, so
    validation log-loss measures generalisation to an unseen season rather than
    to unseen games of a season the model has already learned.
    """
    seasons = _latest_seasons(df)
    if len(seasons) < 2:
        raise ValueError("need at least two seasons for a train/test split")
    test_season = seasons[-1] if test_season is None else test_season

    x_all = df[feature_cols].to_numpy(dtype=np.float64)
    y_all = df[LABEL_COL].to_numpy(dtype=np.float64)
    groups = df[GROUP_COL].to_numpy()

    test_mask = df[GROUP_COL].to_numpy() == test_season
    x_pool, y_pool, groups_pool = x_all[~test_mask], y_all[~test_mask], groups[~test_mask]

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=split_seed)
    train_idx, val_idx = next(splitter.split(x_pool, y_pool, groups=groups_pool))

    return PregameData(
        x_train=x_pool[train_idx],
        y_train=y_pool[train_idx],
        x_val=x_pool[val_idx],
        y_val=y_pool[val_idx],
        x_test=x_all[test_mask],
        y_test=y_all[test_mask],
        feature_cols=feature_cols,
        test_season=int(test_season),
    )


def prepare_chronological_split(
    df: pd.DataFrame, feature_cols: list[str], test_season: int | None = None
) -> PregameData:
    """Train on all but the last two seasons, validate on the penultimate, test on the last.

    Used by XGBoost. Missing values are filled with 0.0, as the notebook did.
    """
    seasons = _latest_seasons(df)
    if len(seasons) < 3:
        raise ValueError("need at least three seasons for a chronological split")
    test_season = seasons[-1] if test_season is None else test_season
    val_season = seasons[seasons.index(test_season) - 1]

    x_all = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y_all = df[LABEL_COL].to_numpy(dtype=np.float64)
    season_values = df[GROUP_COL].to_numpy()

    test_mask = season_values == test_season
    val_mask = season_values == val_season
    train_mask = np.isin(season_values, seasons[: seasons.index(test_season) - 1])

    return PregameData(
        x_train=x_all[train_mask],
        y_train=y_all[train_mask],
        x_val=x_all[val_mask],
        y_val=y_all[val_mask],
        x_test=x_all[test_mask],
        y_test=y_all[test_mask],
        feature_cols=feature_cols,
        test_season=int(test_season),
    )


# --------------------------------------------------------------------------- #
# Logistic regression baseline
# --------------------------------------------------------------------------- #


def train_logreg(data: PregameData) -> tuple[Pipeline, dict[str, ClassificationMetrics]]:
    """L2 logistic regression on standardised features.

    Nearly as accurate as the MLP and the best calibrated of the three, which is
    the project's main result about model capacity.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            # L2 is scikit-learn's default; passing penalty="l2" explicitly is deprecated
            # as of 1.8 and changes nothing, so it is omitted.
            ("logreg", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")),
        ]
    )
    pipeline.fit(data.x_train, data.y_train)

    metrics = {}
    for name in ("train", "val", "test"):
        x, y = data.split(name)
        metrics[name] = classification_metrics(y, pipeline.predict_proba(x)[:, 1])
    return pipeline, metrics


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MlpConfig:
    """The grid search's winning configuration, as the default."""

    hidden_sizes: tuple[int, ...] = (128, 64)
    activation: str = "relu"
    dropout: float = 0.2
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 40
    patience: int = 6
    batch_size: int = 512
    eval_batch_size: int = 1024
    # Early stopping needs this much improvement in val log-loss to count.
    min_delta: float = 1e-4


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
}


def get_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "leaky_relu":
        return nn.LeakyReLU(0.01)
    if key not in _ACTIVATIONS:
        raise ValueError(f"unknown activation: {name}")
    return _ACTIVATIONS[key]()


def build_mlp(input_dim: int, config: MlpConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden in config.hidden_sizes:
        layers.append(nn.Linear(prev_dim, hidden))
        layers.append(get_activation(config.activation))
        if config.dropout > 0:
            layers.append(nn.Dropout(config.dropout))
        prev_dim = hidden
    layers.append(nn.Linear(prev_dim, 1))
    return nn.Sequential(*layers)


@dataclass
class MlpTrainingHistory:
    epochs: list[int] = field(default_factory=list)
    train_bce: list[float] = field(default_factory=list)
    val_log_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False


def _predict_probs(model: nn.Module, x: NDArray[np.float64]) -> NDArray[np.float64]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32)).squeeze(1)
        return torch.sigmoid(logits).numpy().astype(np.float64)


def train_mlp(
    data: PregameData, config: MlpConfig | None = None, seed: int = 0
) -> tuple[nn.Module, StandardScaler, MlpTrainingHistory, dict[str, ClassificationMetrics]]:
    """Train the MLP with early stopping on validation log-loss.

    Runs on CPU with every RNG seeded. The notebooks seeded nothing, so their
    exact numbers are not reachable; this makes *these* numbers reproducible,
    which is what matters for a service that reports a model version.

    Pinned to one thread. On macOS, torch's OpenMP runtime and xgboost's are
    separate copies, and a multi-threaded torch pool poisons any xgboost ``fit``
    later in the same process (see ``nba_winprob.models.__init__``). It also
    removes thread-scheduling nondeterminism from the results, and at 4,506
    training rows by 76 features it costs nothing measurable.
    """
    config = config or MlpConfig()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(data.x_train)
    x_val = scaler.transform(data.x_val)
    x_test = scaler.transform(data.x_test)

    model = build_mlp(x_train.shape[1], config)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(data.y_train, dtype=torch.float32).unsqueeze(1),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )

    history = MlpTrainingHistory()
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(1), yb.squeeze(1))
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        val_metrics = classification_metrics(data.y_val, _predict_probs(model, x_val))
        history.epochs.append(epoch)
        history.train_bce.append(float(np.mean(batch_losses)))
        history.val_log_loss.append(val_metrics["log_loss"])

        if val_metrics["log_loss"] < best_val - config.min_delta:
            best_val = val_metrics["log_loss"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            history.best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                history.stopped_early = True
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "train": classification_metrics(data.y_train, _predict_probs(model, x_train)),
        "val": classification_metrics(data.y_val, _predict_probs(model, x_val)),
        "test": classification_metrics(data.y_test, _predict_probs(model, x_test)),
    }
    return model, scaler, history, metrics


# --------------------------------------------------------------------------- #
# XGBoost
# --------------------------------------------------------------------------- #


def xgb_params(y_train: NDArray[np.float64], seed: int = NOTEBOOK_SPLIT_SEED) -> dict[str, object]:
    """The grid search's winning parameters.

    ``scale_pos_weight`` is the train split's negative/positive ratio (~0.776),
    which down-weights the majority home-win class.
    """
    positives = float(y_train.sum())
    if positives == 0:
        raise ValueError("training split has no positive labels")
    return {
        "n_estimators": 30,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "gamma": 0.1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": seed,
        "scale_pos_weight": (len(y_train) - positives) / positives,
    }


def train_xgboost(
    data: PregameData, seed: int = NOTEBOOK_SPLIT_SEED
) -> tuple[XGBClassifier, dict[str, ClassificationMetrics]]:
    """Train XGBoost for the full 30 rounds, with no early stopping.

    The notebook's ``early_stopping_rounds`` argument raised ``TypeError`` on its
    xgboost version and fell through to a plain fit, so all 30 rounds were used.
    Reproducing the published numbers means doing the same deliberately.

    xgboost is imported here rather than at module scope because its native
    library needs OpenMP (``brew install libomp`` on macOS); the logistic and MLP
    models, and anything serving them, must not fail to import without it.
    """
    from xgboost import XGBClassifier

    model = XGBClassifier(**xgb_params(data.y_train, seed=seed))
    model.fit(
        data.x_train,
        data.y_train,
        eval_set=[(data.x_train, data.y_train), (data.x_val, data.y_val)],
        verbose=False,
    )

    metrics = {}
    for name in ("train", "val", "test"):
        x, y = data.split(name)
        metrics[name] = classification_metrics(y, model.predict_proba(x)[:, 1])
    return model, metrics
