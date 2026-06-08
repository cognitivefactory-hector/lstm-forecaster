"""Time-ordered train/val/test splitting.

The first leakage guard: split by time *before* any scaling or windowing, into
contiguous, ordered segments. Never shuffle, never sample at random — the
forecast origin must always lie strictly between train and what follows it.
"""

from __future__ import annotations

import numpy as np


def time_split(
    series: np.ndarray,
    *,
    val_frac: float,
    test_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a series into contiguous, time-ordered train/val/test segments.

    Args:
        series: 1-D array in time order.
        val_frac: Fraction of the series reserved for validation (the middle).
        test_frac: Fraction reserved for test (the most recent tail).

    Returns:
        ``(train, val, test)`` — contiguous slices whose concatenation, in
        order, reconstructs ``series``. Train is the earliest segment.

    Raises:
        ValueError: if the fractions leave no training data.
    """
    series = np.asarray(series, dtype=float).ravel()
    if val_frac + test_frac >= 1.0:
        raise ValueError(
            f"val_frac + test_frac = {val_frac + test_frac} leaves no training data"
        )

    n = len(series)
    n_val = int(round(n * val_frac))
    n_test = int(round(n * test_frac))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("fractions leave no training data")

    train = series[:n_train]
    val = series[n_train : n_train + n_val]
    test = series[n_train + n_val :]
    return train, val, test
