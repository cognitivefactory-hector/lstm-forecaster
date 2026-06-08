"""Residual-based prediction intervals.

Built from the walk-forward backtest's own per-horizon residuals: take the
empirical quantiles of ``y_true - y_pred`` at each horizon and use them as
offsets around a new forecast. No distributional assumption — if the errors are
skewed or fat-tailed, the interval reflects that, and it naturally widens at
horizons where the model is less certain.
"""

from __future__ import annotations

import numpy as np


def residual_offsets(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    coverage: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-horizon residual quantile offsets for a central prediction interval.

    Args:
        y_true: Backtest actuals, shape ``(n_folds, horizon)``.
        y_pred: Backtest forecasts, same shape.
        coverage: Target central probability mass (e.g. 0.9 -> 5th/95th pct).

    Returns:
        ``(lower, upper)`` offset arrays, one value per horizon. Add them to a
        forecast (see :func:`apply_intervals`) to get the interval bands.

    Raises:
        ValueError: if ``coverage`` is not in ``(0, 1)``.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must be in (0, 1), got {coverage}")

    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    tail = (1.0 - coverage) / 2.0
    lower = np.quantile(residuals, tail, axis=0)
    upper = np.quantile(residuals, 1.0 - tail, axis=0)
    return lower, upper


def apply_intervals(
    forecast: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Offset a forecast by the residual quantiles to get interval bands.

    Returns:
        ``(lower_band, upper_band)`` = ``forecast + lower`` and
        ``forecast + upper``.
    """
    forecast = np.asarray(forecast, dtype=float)
    return forecast + np.asarray(lower, dtype=float), forecast + np.asarray(
        upper, dtype=float
    )
