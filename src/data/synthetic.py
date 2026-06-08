"""Seeded synthetic seasonal-demand generator.

Fully reproducible and obviously synthetic — a known-predictable series for
exercising the leak-free harness and as a sanity target for the models. The
deep model earning its place on *this* series proves nothing about real data;
that's the point of always also testing on a real public series.
"""

from __future__ import annotations

import numpy as np


def make_seasonal_demand(
    n: int,
    *,
    period: int = 7,
    seed: int,
    base: float = 50.0,
    trend: float = 0.01,
    seasonal_amp: float = 10.0,
    noise: float = 1.0,
    period2: int | None = None,
    seasonal_amp2: float = 0.0,
    ar: float = 0.0,
    shock_prob: float = 0.0,
    shock_scale: float = 0.0,
) -> np.ndarray:
    """Generate a seasonal-demand series: trend + seasonality + noise, clipped at 0.

    The optional arguments add structure a *single-lag* baseline cannot capture —
    a second seasonality, autocorrelated noise, and occasional shocks (SPEC §5) —
    giving a deep model legitimate room to add value. All default to off, so the
    basic trend + one-season + white-noise behavior is unchanged.

    Args:
        n: Number of time steps.
        period: Primary seasonal period (e.g. 7 for weekly).
        seed: RNG seed — same seed yields the identical series.
        base: Baseline demand level.
        trend: Per-step linear drift added to the level.
        seasonal_amp: Amplitude of the primary sinusoidal seasonal component.
        noise: Standard deviation of the (innovation) Gaussian noise.
        period2: Optional second seasonal period (e.g. 30 for monthly).
        seasonal_amp2: Amplitude of the second seasonal component.
        ar: AR(1) coefficient applied to the noise (0 = white noise).
        shock_prob: Per-step probability of an additive shock.
        shock_scale: Standard deviation of the shock magnitude.

    Returns:
        A 1-D float array of length ``n``, with all values >= 0.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    level = base + trend * t
    seasonal = seasonal_amp * np.sin(2 * np.pi * t / period)
    if period2:
        seasonal = seasonal + seasonal_amp2 * np.sin(2 * np.pi * t / period2)

    innovations = rng.normal(0.0, noise, size=n)
    if ar:
        noise_series = np.empty(n)
        prev = 0.0
        for i in range(n):
            prev = ar * prev + innovations[i]
            noise_series[i] = prev
    else:
        noise_series = innovations

    series = level + seasonal + noise_series
    if shock_prob > 0:
        fires = rng.random(n) < shock_prob
        series = series + fires * rng.normal(0.0, shock_scale, size=n)

    return np.clip(series, 0.0, None)
