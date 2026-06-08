"""Tests for residual-based prediction intervals — M3.

Intervals come from the backtest's own per-horizon residuals: empirical, honest,
and they widen with the horizon exactly when the model gets less certain. No
Gaussian assumption is imposed on the errors.
"""

import numpy as np
import pytest

from src.eval.intervals import apply_intervals, residual_offsets


def _residuals(n=400, horizon=3, seed=0):
    rng = np.random.default_rng(seed)
    y_pred = rng.normal(size=(n, horizon))
    # Spread grows with the horizon: column h has std (h + 1).
    y_true = y_pred + rng.normal(scale=np.arange(1, horizon + 1), size=(n, horizon))
    return y_true, y_pred


def test_offsets_have_one_value_per_horizon():
    y_true, y_pred = _residuals(horizon=4)
    lower, upper = residual_offsets(y_true, y_pred, coverage=0.9)
    assert lower.shape == (4,)
    assert upper.shape == (4,)


def test_lower_offset_never_exceeds_upper():
    y_true, y_pred = _residuals()
    lower, upper = residual_offsets(y_true, y_pred, coverage=0.8)
    assert np.all(lower <= upper)


def test_empirical_coverage_matches_the_requested_level():
    y_true, y_pred = _residuals(n=2000, horizon=3, seed=1)
    coverage = 0.8
    lower, upper = residual_offsets(y_true, y_pred, coverage=coverage)
    residuals = y_true - y_pred
    inside = (residuals >= lower) & (residuals <= upper)
    np.testing.assert_allclose(inside.mean(axis=0), coverage, atol=0.03)


def test_intervals_widen_with_the_horizon_when_uncertainty_grows():
    y_true, y_pred = _residuals(n=2000, horizon=3, seed=2)
    lower, upper = residual_offsets(y_true, y_pred, coverage=0.9)
    width = upper - lower
    assert np.all(np.diff(width) > 0)


def test_higher_coverage_gives_wider_intervals():
    y_true, y_pred = _residuals(n=2000, seed=3)
    w80 = np.subtract(*reversed(residual_offsets(y_true, y_pred, coverage=0.8)))
    w95 = np.subtract(*reversed(residual_offsets(y_true, y_pred, coverage=0.95)))
    assert np.all(w95 >= w80)


def test_apply_intervals_centers_bands_on_the_forecast():
    forecast = np.array([10.0, 20.0, 30.0])
    lower = np.array([-1.0, -2.0, -3.0])
    upper = np.array([1.0, 2.0, 3.0])
    lo_band, hi_band = apply_intervals(forecast, lower, upper)
    np.testing.assert_array_equal(lo_band, [9.0, 18.0, 27.0])
    np.testing.assert_array_equal(hi_band, [11.0, 22.0, 33.0])


def test_rejects_coverage_outside_the_unit_interval():
    y_true, y_pred = _residuals()
    with pytest.raises(ValueError):
        residual_offsets(y_true, y_pred, coverage=1.5)
