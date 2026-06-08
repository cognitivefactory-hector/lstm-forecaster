"""Tests for the naive and seasonal-naive baselines (the bar) — M2.

These are non-negotiable comparators: the LSTM only earns its place if it beats
seasonal-naive on MASE per horizon. They must be dead simple and exactly right.
"""

import numpy as np
import pytest

from src.models.baselines import naive_forecast, seasonal_naive_forecast


def test_naive_repeats_the_last_observation():
    history = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    np.testing.assert_array_equal(naive_forecast(history, horizon=3), [5.0, 5.0, 5.0])


def test_naive_forecast_has_horizon_length():
    history = np.array([1.0, 2.0, 3.0])
    assert naive_forecast(history, horizon=7).shape == (7,)


def test_seasonal_naive_repeats_the_last_season_within_one_period():
    # period 4: forecast for the next 4 steps is the most recent full season.
    history = np.array([10.0, 20.0, 30.0, 40.0, 11.0, 21.0, 31.0, 41.0])
    out = seasonal_naive_forecast(history, horizon=4, period=4)
    np.testing.assert_array_equal(out, [11.0, 21.0, 31.0, 41.0])


def test_seasonal_naive_wraps_the_last_season_beyond_one_period():
    history = np.array([10.0, 20.0, 30.0, 40.0, 11.0, 21.0, 31.0, 41.0])
    out = seasonal_naive_forecast(history, horizon=6, period=4)
    # steps 5 and 6 wrap back to the start of the most recent season.
    np.testing.assert_array_equal(out, [11.0, 21.0, 31.0, 41.0, 11.0, 21.0])


def test_seasonal_naive_with_period_one_equals_naive():
    history = np.array([1.0, 2.0, 9.0])
    np.testing.assert_array_equal(
        seasonal_naive_forecast(history, horizon=3, period=1),
        naive_forecast(history, horizon=3),
    )


def test_seasonal_naive_needs_a_full_season_of_history():
    history = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        seasonal_naive_forecast(history, horizon=2, period=4)
