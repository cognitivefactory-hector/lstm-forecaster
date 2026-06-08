"""Demo datasets: seeded synthetic demand and AAPL realized volatility.

The financial dataset deliberately forecasts *volatility*, not price. Price level
is ~a random walk (whiteboard Q3) where naive is unbeatable for the right
reasons; realized volatility is persistent and genuinely more forecastable, so
it's the honest financial variant — framed as method, never a market-beating
claim. A small real AAPL sample is committed so the demo runs fully offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.synthetic import make_seasonal_demand

_SAMPLE = Path(__file__).parent / "samples" / "aapl_close.csv"

DISCLAIMER = (
    "Realized volatility of AAPL daily log returns (public sample, 2022-2024). "
    "Forecasts volatility, not price level. Not investment advice."
)


@dataclass
class DatasetSpec:
    """A demo series plus the seasonal period its baseline/MASE uses."""

    name: str
    label: str
    series: np.ndarray
    season: int
    note: str


def load_aapl_close(path: Path = _SAMPLE) -> np.ndarray:
    """Load the committed AAPL daily-close sample as a 1-D float array."""
    frame = pd.read_csv(path)
    closes = pd.to_numeric(frame.iloc[:, 1], errors="coerce").dropna()
    return closes.to_numpy(dtype=float)


def realized_volatility(prices: np.ndarray, window: int = 10) -> np.ndarray:
    """Rolling standard deviation of daily log returns (a realized-vol proxy).

    Returns a 1-D array of length ``len(prices) - window`` (one differencing
    step plus the rolling warm-up dropped), all non-negative.
    """
    prices = np.asarray(prices, dtype=float).ravel()
    log_returns = np.diff(np.log(prices))
    vol = pd.Series(log_returns).rolling(window).std().dropna()
    return vol.to_numpy(dtype=float)


def load_synthetic() -> DatasetSpec:
    # Weekly + monthly seasonality, AR(1) noise, and occasional shocks: structure
    # a single-lag seasonal-naive cannot capture but an LSTM can. This is the
    # series where the deep model legitimately earns its place (it is not rigged —
    # the win comes from real, learnable structure the baseline ignores).
    return DatasetSpec(
        name="synthetic_demand",
        label="Synthetic demand (weekly + monthly + AR + shocks)",
        series=make_seasonal_demand(
            n=800,
            period=7,
            seed=0,
            base=60.0,
            trend=0.02,
            seasonal_amp=6.0,
            noise=4.0,
            period2=30,
            seasonal_amp2=11.0,
            ar=0.45,
            shock_prob=0.03,
            shock_scale=12.0,
        ),
        season=7,
        note="Seeded: trend + weekly & monthly seasonality + AR(1) noise + shocks.",
    )


def load_financial() -> DatasetSpec:
    return DatasetSpec(
        name="aapl_volatility",
        label="AAPL realized volatility (public sample)",
        series=realized_volatility(load_aapl_close(), window=10),
        season=1,  # volatility is persistent; naive persistence is the bar
        note=DISCLAIMER,
    )


_LOADERS = {
    "synthetic_demand": load_synthetic,
    "aapl_volatility": load_financial,
}


def dataset_names() -> list[str]:
    """Names of the available demo datasets."""
    return list(_LOADERS)


def get_dataset(name: str) -> DatasetSpec:
    """Build the named dataset.

    Raises:
        ValueError: if ``name`` is not a known dataset.
    """
    if name not in _LOADERS:
        raise ValueError(f"unknown dataset {name!r} (have {dataset_names()})")
    return _LOADERS[name]()
