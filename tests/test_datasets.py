"""Tests for the demo dataset registry — M6.

Two datasets: the always-available seeded synthetic demand series, and a real
financial series (AAPL) turned into *realized volatility* — because forecasting
price level is a random walk (whiteboard Q3), but volatility is persistent and
genuinely more forecastable. Each carries the seasonal period its baseline uses.
"""

import numpy as np
import pytest

from app.datasets import (
    dataset_names,
    get_dataset,
    load_aapl_close,
    realized_volatility,
)


def test_realized_volatility_is_non_negative_and_shorter_than_prices():
    prices = np.array([100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0, 96.0, 105.0])
    vol = realized_volatility(prices, window=3)
    assert vol.ndim == 1
    assert len(vol) == len(prices) - 3  # one diff + (window-1) NaNs dropped
    assert np.all(vol >= 0)
    assert np.all(np.isfinite(vol))


def test_aapl_sample_loads_as_positive_prices():
    prices = load_aapl_close()
    assert prices.ndim == 1
    assert len(prices) > 100  # the committed sample has ~500 trading days
    assert np.all(prices > 0)


def test_synthetic_dataset_has_weekly_season():
    spec = get_dataset("synthetic_demand")
    assert spec.season == 7
    assert spec.series.ndim == 1
    assert len(spec.series) > 100


def test_financial_dataset_is_persistent_volatility_with_season_one():
    # Volatility's honest baseline is naive persistence (season=1), not a
    # weekly seasonal cycle.
    spec = get_dataset("aapl_volatility")
    assert spec.season == 1
    assert np.all(spec.series >= 0)
    assert "not investment advice" in spec.note.lower()


def test_dataset_names_lists_both_datasets():
    names = dataset_names()
    assert "synthetic_demand" in names
    assert "aapl_volatility" in names


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        get_dataset("nonexistent")
