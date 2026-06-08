"""Tests for the walk-forward (rolling-origin) backtest engine — M3.

This is where the leak-free discipline becomes an *evaluation protocol*: every
fold trains only on data strictly before its forecast origin, and the model is
refit per fold. The engine is model-agnostic — baselines now, the LSTM later,
through the exact same harness.
"""

import numpy as np
import pytest

from src.eval.metrics import mae, per_horizon
from src.eval.walk_forward import walk_forward
from src.models.baselines import naive_forecast


def _naive(train, horizon):
    return naive_forecast(train, horizon)


def test_each_fold_trains_only_on_data_before_its_origin():
    series = np.arange(20, dtype=float)
    result = walk_forward(series, _naive, horizon=3, initial_train=10, step=1)
    for fold in result.folds:
        # Train is exactly the prefix up to the origin; actuals start at origin.
        np.testing.assert_array_equal(fold.train, series[: fold.origin])
        np.testing.assert_array_equal(
            fold.actual, series[fold.origin : fold.origin + 3]
        )
        assert fold.train.max() < fold.actual.min()  # no future leakage


def test_forecaster_never_receives_future_data():
    series = np.arange(30, dtype=float)
    seen_max = []

    def spy(train, horizon):
        seen_max.append(train.max())
        return naive_forecast(train, horizon)

    result = walk_forward(series, spy, horizon=4, initial_train=12, step=2)
    for fold, train_max in zip(result.folds, seen_max, strict=True):
        # The largest value the forecaster saw is the step just before origin.
        assert train_max == series[fold.origin - 1]


def test_number_of_folds_follows_origin_and_step():
    series = np.arange(20, dtype=float)
    # origins 10,12,14,16 are valid (origin+horizon<=20); 18 would overflow.
    result = walk_forward(series, _naive, horizon=2, initial_train=10, step=2)
    assert len(result.folds) == 5  # origins 10,12,14,16,18


def test_stacked_arrays_have_shape_n_folds_by_horizon():
    series = np.arange(20, dtype=float)
    result = walk_forward(series, _naive, horizon=3, initial_train=10, step=1)
    assert result.y_true.shape == (len(result.folds), 3)
    assert result.y_pred.shape == (len(result.folds), 3)


def test_horizon_error_increases_monotonically_on_a_ramp():
    # On a linear ramp, a naive ("repeat last") forecast errs by exactly k at
    # horizon step k -> per-horizon MAE is strictly increasing. The classic
    # signature of compounding multistep error, made visible.
    series = np.arange(100, dtype=float)
    result = walk_forward(series, _naive, horizon=5, initial_train=20, step=1)
    per_h = per_horizon(mae, result.y_true, result.y_pred)
    assert np.all(np.diff(per_h) > 0)
    np.testing.assert_allclose(per_h, [1, 2, 3, 4, 5])


def test_sliding_window_keeps_a_fixed_train_length():
    series = np.arange(30, dtype=float)
    result = walk_forward(
        series, _naive, horizon=2, initial_train=10, step=3, expanding=False
    )
    for fold in result.folds:
        assert len(fold.train) == 10  # fixed window, not growing


def test_expanding_window_grows_the_train_length():
    series = np.arange(30, dtype=float)
    result = walk_forward(
        series, _naive, horizon=2, initial_train=10, step=3, expanding=True
    )
    lengths = [len(f.train) for f in result.folds]
    assert lengths == sorted(lengths) and lengths[0] < lengths[-1]


def test_raises_when_series_too_short_for_one_fold():
    series = np.arange(10, dtype=float)
    with pytest.raises(ValueError):
        walk_forward(series, _naive, horizon=5, initial_train=8, step=1)
