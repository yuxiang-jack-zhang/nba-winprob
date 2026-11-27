"""Unit tests for the metrics module, on synthetic data with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from nba_winprob.models.metrics import (
    brier_score,
    calibration_bins,
    classification_metrics,
    expected_calibration_error,
    log_loss_manual,
    metrics_by_group,
    sharpness,
)


def test_brier_score_known_values() -> None:
    assert brier_score([1, 0], [1.0, 0.0]) == 0.0
    assert brier_score([1, 0], [0.0, 1.0]) == 1.0
    assert brier_score([1, 0], [0.5, 0.5]) == pytest.approx(0.25)


def test_log_loss_manual_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    y_true = (rng.random(500) > 0.5).astype(float)
    y_prob = rng.uniform(0.02, 0.98, 500)
    assert log_loss_manual(y_true, y_prob) == pytest.approx(
        classification_metrics(y_true, y_prob)["log_loss"], abs=1e-9
    )


def test_log_loss_manual_clips_instead_of_diverging() -> None:
    # A confident wrong prediction must give a large finite loss, not inf.
    value = log_loss_manual([1.0], [0.0])
    assert np.isfinite(value) and value > 30


def test_sharpness_is_zero_for_a_constant_forecast() -> None:
    # The "always predict the base rate" model: perfectly calibrated, useless.
    assert sharpness([0.5] * 100) == 0.0


def test_ece_is_zero_for_a_perfectly_calibrated_forecast() -> None:
    # 100 samples at p=0.75 with exactly 75 positives: the bin's confidence and
    # its observed rate agree, so there is no calibration error to find.
    y_prob = np.full(100, 0.75)
    y_true = np.concatenate([np.ones(75), np.zeros(25)])
    assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.0, abs=1e-12)


def test_ece_finds_a_miscalibrated_forecast() -> None:
    y_prob = np.full(100, 0.9)
    y_true = np.concatenate([np.ones(50), np.zeros(50)])
    assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.4, abs=1e-12)


def test_calibration_bins_keeps_empty_bins() -> None:
    ece, bins = calibration_bins([1, 0], [0.75, 0.75], n_bins=10)
    assert len(bins) == 10
    assert int(bins["count"].sum()) == 2
    # Only the 0.7-0.8 bin is populated; the rest are NaN rows, so a plot shows gaps.
    assert bins["avg_conf"].isna().sum() == 9
    assert ece == pytest.approx(0.25)


def test_classification_metrics_shape_and_accuracy() -> None:
    y_true = [1, 1, 0, 0]
    y_prob = [0.9, 0.6, 0.4, 0.1]
    metrics = classification_metrics(y_true, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["n"] == 4


def test_metrics_mismatched_shapes_raise() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        calibration_bins([1, 0], [0.5])


def test_metrics_by_group_splits_rows() -> None:
    y_true = [1, 0, 1, 0]
    y_prob = [0.9, 0.1, 0.2, 0.8]
    table = metrics_by_group(y_true, y_prob, ["early", "early", "late", "late"])
    assert set(table["group"]) == {"early", "late"}
    assert table.set_index("group").loc["early", "accuracy"] == 1.0
    assert table.set_index("group").loc["late", "accuracy"] == 0.0


def test_metrics_by_group_handles_single_class_group() -> None:
    # Late-game states can be all-wins; AUC is undefined there but accuracy is not,
    # so the row must still appear rather than blowing up the whole breakdown.
    table = metrics_by_group([1, 1, 0], [0.9, 0.8, 0.2], ["a", "a", "b"])
    row = table.set_index("group").loc["a"]
    assert row["accuracy"] == 1.0
    assert np.isnan(row["roc_auc"])
