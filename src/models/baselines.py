"""Naive and seasonal-naive baselines — the bar the LSTM must clear.

Built first and always shown. The headline of the project is not "the LSTM
wins" but "here is whether the deep model's complexity is justified versus one
line of forecasting." Seasonal-naive is the reference MASE is scaled against.
"""

from __future__ import annotations

import numpy as np


def naive_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast every future step as the last observed value ("tomorrow = today")."""
    history = np.asarray(history, dtype=float).ravel()
    return np.full(horizon, history[-1], dtype=float)


def seasonal_naive_forecast(
    history: np.ndarray, horizon: int, period: int
) -> np.ndarray:
    """Forecast step ``h`` as the observation one season (``period``) earlier.

    Repeats the most recent full season, wrapping when the horizon exceeds one
    period. ``period=1`` reduces to :func:`naive_forecast`.

    Raises:
        ValueError: if ``history`` is shorter than one full season.
    """
    history = np.asarray(history, dtype=float).ravel()
    if len(history) < period:
        raise ValueError(
            f"history of length {len(history)} shorter than one season (period={period})"
        )
    last_season = history[-period:]
    return np.array([last_season[h % period] for h in range(horizon)], dtype=float)
