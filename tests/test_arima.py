"""Tests for the ARIMA baseline with AIC-based order selection — M2.

ARIMA is the second mandatory comparator. We keep the grid tiny and the series
small so this stays fast; the contract is correct shapes and that order
selection actually minimizes AIC.
"""

import numpy as np
import pytest

from src.models.arima import arima_forecast, select_order


def _ar1_series(n=150, phi=0.8, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    return x


def test_forecast_has_horizon_length():
    train = _ar1_series()
    forecast, order = arima_forecast(train, horizon=5, candidate_orders=[(1, 0, 0)])
    assert forecast.shape == (5,)
    assert order == (1, 0, 0)


def test_select_order_minimizes_aic_over_the_grid():
    # On a strongly autocorrelated AR(1) series, AR(1) must beat white noise.
    train = _ar1_series(phi=0.85, seed=1)
    order = select_order(train, candidate_orders=[(0, 0, 0), (1, 0, 0)])
    assert order == (1, 0, 0)


def test_select_order_returns_a_candidate_from_the_grid():
    train = _ar1_series(seed=2)
    grid = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    assert select_order(train, candidate_orders=grid) in grid


def test_forecast_selects_order_when_grid_given():
    train = _ar1_series(phi=0.85, seed=3)
    forecast, order = arima_forecast(
        train, horizon=3, candidate_orders=[(0, 0, 0), (1, 0, 0)]
    )
    assert forecast.shape == (3,)
    assert order == (1, 0, 0)


def test_empty_grid_raises():
    with pytest.raises(ValueError):
        select_order(_ar1_series(), candidate_orders=[])
