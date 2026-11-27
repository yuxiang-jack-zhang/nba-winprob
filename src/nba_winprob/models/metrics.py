"""Probabilistic classification metrics.

One implementation, replacing the two divergent copies in the notebooks: the MLP
notebook returned calibration bins as a dict, the XGBoost notebook as a DataFrame,
with the same arithmetic written twice. The binning convention is kept exactly as
the originals had it so the reproduced numbers are comparable.

A win-probability model is judged on calibration, not just accuracy -- "the home
team wins 70% of the time" is only useful if those games really are won 70% of the
time -- so Brier, ECE and sharpness are reported alongside log-loss and AUC.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import log_loss, roc_auc_score

DEFAULT_N_BINS = 10


class ClassificationMetrics(TypedDict):
    """Scalar metrics for one split. Keys match the notebooks' result dicts."""

    log_loss: float
    accuracy: float
    roc_auc: float
    brier: float
    sharpness: float
    ece: float
    n: int


def _as_float(values: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


def brier_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Mean squared error of the probability forecast. Lower is better."""
    return float(np.mean((_as_float(y_prob) - _as_float(y_true)) ** 2))


def log_loss_manual(y_true: ArrayLike, y_prob: ArrayLike, eps: float = 1e-15) -> float:
    """Pure-numpy log-loss, kept as a cross-check on sklearn's."""
    y_true_arr = _as_float(y_true)
    y_prob_arr = np.clip(_as_float(y_prob), eps, 1 - eps)
    return float(
        -np.mean(y_true_arr * np.log(y_prob_arr) + (1 - y_true_arr) * np.log(1 - y_prob_arr))
    )


def sharpness(y_prob: ArrayLike) -> float:
    """Spread of the predicted probabilities.

    A model that always predicts the base rate is perfectly calibrated and
    useless; sharpness is what separates the two, so it is reported next to ECE.
    """
    return float(np.std(_as_float(y_prob)))


def calibration_bins(
    y_true: ArrayLike, y_prob: ArrayLike, n_bins: int = DEFAULT_N_BINS
) -> tuple[float, pd.DataFrame]:
    """Equal-width calibration bins over [0, 1] and the resulting ECE.

    Returns ``(ece, bins)`` where ``bins`` has one row per bin with its count,
    mean predicted probability and observed win rate. Empty bins are kept as rows
    with NaN so a calibration plot shows the gaps.

    Binning follows the notebooks: ``np.digitize(..., right=True)`` over
    ``n_bins + 1`` edges, taking bins ``1..n_bins``. A probability of exactly 0.0
    falls outside that range and is excluded, which is harmless for models that
    never predict a hard zero and is preserved for comparability.
    """
    y_true_arr = _as_float(y_true)
    y_prob_arr = _as_float(y_prob)
    if y_true_arr.shape != y_prob_arr.shape:
        raise ValueError(f"shape mismatch: {y_true_arr.shape} vs {y_prob_arr.shape}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob_arr, edges, right=True)

    total = len(y_true_arr)
    ece = 0.0
    rows: list[dict[str, float | int]] = []
    for b in range(1, n_bins + 1):
        mask = bin_ids == b
        count = int(mask.sum())
        row: dict[str, float | int] = {
            "bin": b,
            "count": count,
            "left_edge": float(edges[b - 1]),
            "right_edge": float(edges[b]),
        }
        if count == 0:
            row["avg_conf"] = float("nan")
            row["avg_acc"] = float("nan")
        else:
            avg_conf = float(y_prob_arr[mask].mean())
            avg_acc = float(y_true_arr[mask].mean())
            row["avg_conf"] = avg_conf
            row["avg_acc"] = avg_acc
            ece += (count / total) * abs(avg_conf - avg_acc)
        rows.append(row)

    columns = ["bin", "count", "avg_conf", "avg_acc", "left_edge", "right_edge"]
    return float(ece), pd.DataFrame(rows)[columns]


def expected_calibration_error(
    y_true: ArrayLike, y_prob: ArrayLike, n_bins: int = DEFAULT_N_BINS
) -> float:
    return calibration_bins(y_true, y_prob, n_bins=n_bins)[0]


def classification_metrics(
    y_true: ArrayLike, y_prob: ArrayLike, n_bins: int = DEFAULT_N_BINS
) -> ClassificationMetrics:
    """Every scalar metric for one split, from true labels and predicted probabilities."""
    y_true_arr = _as_float(y_true)
    y_prob_arr = _as_float(y_prob)
    return ClassificationMetrics(
        log_loss=float(log_loss(y_true_arr, y_prob_arr, labels=[0, 1])),
        accuracy=float(((y_prob_arr >= 0.5) == (y_true_arr >= 0.5)).mean()),
        roc_auc=float(roc_auc_score(y_true_arr, y_prob_arr)),
        brier=brier_score(y_true_arr, y_prob_arr),
        sharpness=sharpness(y_prob_arr),
        ece=expected_calibration_error(y_true_arr, y_prob_arr, n_bins=n_bins),
        n=len(y_true_arr),
    )


def metrics_by_group(
    y_true: ArrayLike, y_prob: ArrayLike, groups: ArrayLike, n_bins: int = DEFAULT_N_BINS
) -> pd.DataFrame:
    """Per-group metrics, one row per group.

    Used for the in-game model's by-stage breakdown (Q1 ... final two minutes),
    where a single pooled accuracy hides that late-game prediction is far easier
    than early-game.
    """
    y_true_arr = _as_float(y_true)
    y_prob_arr = _as_float(y_prob)
    group_arr = np.asarray(groups)

    rows: list[dict[str, object]] = []
    for group in pd.unique(group_arr):
        mask = group_arr == group
        if not mask.any():
            continue
        subset_true = y_true_arr[mask]
        # AUC and log-loss are undefined for a single-class group; report what we can.
        if len(np.unique(subset_true)) < 2:
            rows.append(
                {
                    "group": group,
                    "n": int(mask.sum()),
                    "accuracy": float(((y_prob_arr[mask] >= 0.5) == (subset_true >= 0.5)).mean()),
                    "brier": brier_score(subset_true, y_prob_arr[mask]),
                    "log_loss": float("nan"),
                    "roc_auc": float("nan"),
                    "sharpness": sharpness(y_prob_arr[mask]),
                    "ece": float("nan"),
                }
            )
            continue
        metrics = classification_metrics(subset_true, y_prob_arr[mask], n_bins=n_bins)
        rows.append({"group": group, **metrics})
    return pd.DataFrame(rows)
