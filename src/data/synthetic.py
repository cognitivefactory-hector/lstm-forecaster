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
) -> np.ndarray:
    """Generate a seasonal-demand series: trend + seasonality + noise, clipped at 0.

    Args:
        n: Number of time steps.
        period: Seasonal period (e.g. 7 for weekly).
        seed: RNG seed — same seed yields the identical series.
        base: Baseline demand level.
        trend: Per-step linear drift added to the level.
        seasonal_amp: Amplitude of the sinusoidal seasonal component.
        noise: Standard deviation of the additive Gaussian noise.

    Returns:
        A 1-D float array of length ``n``, with all values >= 0.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    level = base + trend * t
    seasonal = seasonal_amp * np.sin(2 * np.pi * t / period)
    series = level + seasonal + rng.normal(0.0, noise, size=n)
    return np.clip(series, 0.0, None)
