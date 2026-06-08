"""Honest scoreboard metrics: MAE, RMSE, and MASE.

MASE (Mean Absolute Scaled Error) is the headline. It divides the forecast's
MAE by the *in-sample* MAE of a seasonal-naive forecast on the training series,
so MASE < 1 means the model beats the naive baseline and MASE >= 1 means it does
not. Being scale-free and baseline-relative, it is hard to flatter yourself
with — which is exactly why it's the headline (SPEC.md §7).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _seasonal_naive_scale(y_train: np.ndarray, season: int) -> float:
    """In-sample MAE of the seasonal-naive forecast — the MASE denominator."""
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= season:
        raise ValueError(
            f"training series of length {len(y_train)} too short for season={season}"
        )
    scale = float(np.mean(np.abs(y_train[season:] - y_train[:-season])))
    if scale == 0.0:
        raise ValueError(
            "seasonal-naive scale is zero (training series is perfectly periodic "
            f"at season={season}); MASE is undefined"
        )
    return scale


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    *,
    season: int = 1,
) -> float:
    """Mean absolute scaled error, scaled by the train seasonal-naive MAE.

    Args:
        y_true: Actual values over the evaluation window.
        y_pred: Forecast values.
        y_train: Training series used to compute the naive scale.
        season: Seasonal lag for the naive baseline (1 = plain naive).

    Raises:
        ValueError: if the training series is too short or perfectly periodic
            at ``season`` (scale would be zero).
    """
    return mae(y_true, y_pred) / _seasonal_naive_scale(y_train, season)


def per_horizon(
    metric: Callable[..., float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Apply a scalar ``metric`` to each horizon column independently.

    ``y_true``/``y_pred`` are shaped ``(n_windows, horizon)``; the result has one
    value per horizon, so degradation across the horizon stays visible rather
    than hidden behind a single aggregate (SPEC.md §7).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    horizon = y_true.shape[1]
    return np.array(
        [metric(y_true[:, h], y_pred[:, h], **kwargs) for h in range(horizon)]
    )
