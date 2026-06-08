"""Sliding-window builder producing multistep (direct) targets.

Each window is ``input_len`` consecutive observations as input and the next
``horizon`` observations as the target. Windows are emitted in strict time
order — never shuffled — and a target step never overlaps its own input.
"""

from __future__ import annotations

import numpy as np


def make_windows(
    series: np.ndarray,
    *,
    input_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding input/target windows for multistep forecasting.

    Args:
        series: 1-D array of observations, in time order.
        input_len: Number of past steps per input window.
        horizon: Number of future steps to predict per window.

    Returns:
        ``(X, y)`` where ``X`` has shape ``(n_windows, input_len)`` and ``y``
        has shape ``(n_windows, horizon)``. ``n_windows`` is
        ``len(series) - input_len - horizon + 1``.

    Raises:
        ValueError: if the series is too short for even one full window.
    """
    series = np.asarray(series, dtype=float).ravel()
    n_windows = len(series) - input_len - horizon + 1
    if n_windows < 1:
        raise ValueError(
            f"series of length {len(series)} too short for input_len={input_len} "
            f"+ horizon={horizon} (needs >= {input_len + horizon})"
        )

    X = np.empty((n_windows, input_len), dtype=float)
    y = np.empty((n_windows, horizon), dtype=float)
    for i in range(n_windows):
        X[i] = series[i : i + input_len]
        y[i] = series[i + input_len : i + input_len + horizon]
    return X, y
