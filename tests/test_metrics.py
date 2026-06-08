"""Tests for the honest scoreboard metrics (MASE / RMSE / MAE) — M2.

MASE is the headline because it is scale-free and baseline-relative: MASE < 1
means you beat the naive forecast, MASE >= 1 means you didn't. The tests pin it
to hand-computed values so the metric can't quietly flatter the model.
"""

import numpy as np
import pytest

from src.eval.metrics import mae, mase, per_horizon, rmse


def test_mae_matches_hand_computation():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 4.0, 3.0, 0.0])  # abs errors: 0, 2, 0, 4 -> mean 1.5
    assert mae(y_true, y_pred) == pytest.approx(1.5)


def test_rmse_matches_hand_computation():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([3.0, 0.0, 4.0])  # sq errors 9, 0, 16 -> mean 25/3 -> sqrt
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(25.0 / 3.0))


def test_mae_of_perfect_forecast_is_zero():
    y = np.array([5.0, 7.0, 9.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0


def test_mase_of_a_naive_forecast_is_about_one():
    # On a unit ramp, the in-sample one-step naive error is always 1, so the
    # MASE denominator is 1. A naive one-step forecast on a test ramp also errs
    # by 1 each step -> MASE == 1 exactly. This is the sanity anchor.
    train = np.arange(50, dtype=float)
    y_true = np.array([50.0, 51.0, 52.0])
    y_pred = np.array([49.0, 50.0, 51.0])  # "tomorrow = today" naive, off by 1
    assert mase(y_true, y_pred, train, season=1) == pytest.approx(1.0)


def test_mase_below_one_means_better_than_naive():
    train = np.arange(50, dtype=float)  # scale denominator = 1.0
    y_true = np.array([50.0, 51.0, 52.0])
    y_pred = np.array([50.0, 51.0, 52.0])  # perfect -> error 0 -> MASE 0
    assert mase(y_true, y_pred, train, season=1) == pytest.approx(0.0)


def test_mase_uses_the_seasonal_scale_when_season_given():
    # Seasonal-naive denominator uses lag = season, not lag 1.
    train = np.array([0.0, 10.0, 0.0, 10.0, 0.0, 10.0])  # period-2 oscillation
    # |train[t] - train[t-2]| is 0 everywhere -> degenerate; guard expects > 0.
    with pytest.raises(ValueError):
        mase(np.array([1.0]), np.array([1.0]), train, season=2)


def test_per_horizon_returns_one_value_per_step():
    # Columns are horizons. Errors per column: h0 -> 0, h1 -> 2.
    y_true = np.array([[1.0, 1.0], [2.0, 2.0]])
    y_pred = np.array([[1.0, 3.0], [2.0, 4.0]])
    result = per_horizon(mae, y_true, y_pred)
    np.testing.assert_allclose(result, [0.0, 2.0])
