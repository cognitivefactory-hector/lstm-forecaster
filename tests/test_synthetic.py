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


# --- Enriched structure (M6): give a deep model room a single-lag baseline lacks ---


def _autocorr(series, lag):
    c = series - series.mean()
    return float(np.corrcoef(c[:-lag], c[lag:])[0, 1])


def test_second_seasonal_component_adds_autocorrelation_at_its_period():
    # A weekly + monthly series is autocorrelated at BOTH 7 and 30; a
    # seasonal-naive at lag 7 alone can't capture the period-30 structure.
    series = make_seasonal_demand(
        n=2000, period=7, seed=0, noise=0.5, period2=30, seasonal_amp2=8.0
    )
    assert _autocorr(series, 30) > _autocorr(series, 15)


def test_ar_noise_is_autocorrelated_at_lag_one():
    # With only AR(1) noise (no seasonality/trend), lag-1 autocorrelation is high
    # — short-term structure an LSTM can exploit but naive/seasonal-naive cannot.
    series = make_seasonal_demand(
        n=2000, period=7, seed=0, base=100.0, trend=0.0,
        seasonal_amp=0.0, noise=1.0, ar=0.7,
    )
    assert _autocorr(series, 1) > 0.4


def test_shocks_change_the_series_but_keep_determinism():
    common = dict(n=400, period=7, seed=5, noise=0.5)
    calm = make_seasonal_demand(**common)
    shocked = make_seasonal_demand(**common, shock_prob=0.05, shock_scale=30.0)
    shocked_again = make_seasonal_demand(**common, shock_prob=0.05, shock_scale=30.0)

    assert not np.array_equal(calm, shocked)  # shocks actually fire
    np.testing.assert_array_equal(shocked, shocked_again)  # still reproducible


def test_enriched_series_stays_non_negative():
    series = make_seasonal_demand(
        n=600, period=7, seed=1, period2=30, seasonal_amp2=8.0,
        ar=0.6, shock_prob=0.04, shock_scale=15.0,
    )
    assert (series >= 0).all()
