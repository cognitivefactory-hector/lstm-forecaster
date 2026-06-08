"""Tests for the seeded synthetic seasonal-demand generator (M1)."""

import numpy as np

from src.data.synthetic import make_seasonal_demand


def test_returns_1d_array_of_requested_length():
    series = make_seasonal_demand(n=300, period=7, seed=0)
    assert series.ndim == 1
    assert series.shape == (300,)


def test_is_reproducible_for_a_given_seed():
    a = make_seasonal_demand(n=200, period=7, seed=42)
    b = make_seasonal_demand(n=200, period=7, seed=42)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_produce_different_series():
    a = make_seasonal_demand(n=200, period=7, seed=1)
    b = make_seasonal_demand(n=200, period=7, seed=2)
    assert not np.array_equal(a, b)


def test_has_seasonality_at_the_requested_period():
    # Autocorrelation at the seasonal lag should exceed a clearly non-seasonal
    # lag, otherwise the "seasonal" generator isn't seasonal.
    period = 7
    series = make_seasonal_demand(n=1000, period=period, seed=3, noise=0.5)
    centered = series - series.mean()

    def autocorr(lag):
        return float(np.corrcoef(centered[:-lag], centered[lag:])[0, 1])

    assert autocorr(period) > autocorr(period // 2)


def test_demand_is_non_negative():
    # Demand can't be negative; the generator must clip the floor.
    series = make_seasonal_demand(n=500, period=7, seed=4, noise=20.0, base=5.0)
    assert (series >= 0).all()
