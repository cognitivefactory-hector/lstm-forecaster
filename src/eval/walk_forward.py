"""Walk-forward (rolling-origin) backtest engine.

The leak-free discipline as an evaluation protocol. The origin rolls forward;
at each fold the model is refit on data strictly before the origin and forecasts
the next ``horizon`` steps. The engine is model-agnostic: it hands the
``forecaster`` only past data, so leak-freeness is structural — the same harness
evaluates the naive/ARIMA baselines and (M4) the LSTM identically.

Any scaling a model needs is the model's own concern: it fits on the train slice
it receives, which is already past-only, so train-only scaling stays leak-free.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Forecaster = Callable[[np.ndarray, int], np.ndarray]


@dataclass
class Fold:
    """One rolling-origin fold."""

    origin: int  # index where train ends and the forecast begins
    train: np.ndarray  # observations strictly before the origin
    actual: np.ndarray  # the horizon actuals from the origin onward
    forecast: np.ndarray  # the model's forecast for those steps


@dataclass
class BacktestResult:
    """All folds plus actuals/forecasts stacked as ``(n_folds, horizon)``."""

    folds: list[Fold]
    y_true: np.ndarray
    y_pred: np.ndarray


def walk_forward(
    series: np.ndarray,
    forecaster: Forecaster,
    *,
    horizon: int,
    initial_train: int,
    step: int = 1,
    expanding: bool = True,
) -> BacktestResult:
    """Backtest ``forecaster`` over ``series`` with a rolling forecast origin.

    Args:
        series: The full series, in time order.
        forecaster: ``(train, horizon) -> forecast`` of length ``horizon``.
            Receives only data before the origin and is called afresh per fold.
        horizon: Steps forecast at each origin.
        initial_train: Index of the first forecast origin (and, for a sliding
            window, the fixed train length).
        step: How far the origin advances between folds.
        expanding: If True, train grows to ``series[:origin]``; if False, a
            sliding window of length ``initial_train`` ending at the origin.

    Returns:
        A :class:`BacktestResult`.

    Raises:
        ValueError: if no full fold fits in the series.
    """
    series = np.asarray(series, dtype=float).ravel()
    n = len(series)
    if initial_train + horizon > n:
        raise ValueError(
            f"series of length {n} too short for initial_train={initial_train} "
            f"+ horizon={horizon}"
        )

    folds: list[Fold] = []
    origin = initial_train
    while origin + horizon <= n:
        start = 0 if expanding else origin - initial_train
        train = series[start:origin]
        actual = series[origin : origin + horizon]
        forecast = np.asarray(forecaster(train, horizon), dtype=float)
        folds.append(
            Fold(origin=origin, train=train, actual=actual, forecast=forecast)
        )
        origin += step

    y_true = np.vstack([f.actual for f in folds])
    y_pred = np.vstack([f.forecast for f in folds])
    return BacktestResult(folds=folds, y_true=y_true, y_pred=y_pred)
