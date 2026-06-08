"""The leak-free data harness — the safety core, built before any model.

Composes the three guards in the only order that is honest:

1. Split by time (``time_split``) — contiguous, ordered, never shuffled.
2. Fit the scaler on train ONLY (``TrainOnlyScaler``), transform every split.
3. Window each split *independently* — so no input/target window ever straddles
   the forecast origin between train and what follows it.

``audit_no_scaler_leakage`` is the loud tripwire: it raises if a scaler's
parameters don't match a train-only fit (e.g. it was fit on the full series).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.data.scaling import TrainOnlyScaler
from src.data.splits import time_split
from src.data.windowing import make_windows


class LeakageError(AssertionError):
    """Raised when a leakage guard is violated."""


@dataclass
class PreparedData:
    """Windowed, train-only-scaled splits ready for modeling/backtesting."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: TrainOnlyScaler


def prepare_splits(
    series: np.ndarray,
    *,
    input_len: int,
    horizon: int,
    val_frac: float,
    test_frac: float,
) -> PreparedData:
    """Run the full leak-free pipeline and return windowed, scaled splits.

    Scaling statistics come only from the training segment; each split is
    windowed on its own so no window crosses the forecast origin.
    """
    train_raw, val_raw, test_raw = time_split(
        series, val_frac=val_frac, test_frac=test_frac
    )

    scaler = TrainOnlyScaler().fit(train_raw)
    train_s = scaler.transform(train_raw)
    val_s = scaler.transform(val_raw)
    test_s = scaler.transform(test_raw)

    X_train, y_train = make_windows(train_s, input_len=input_len, horizon=horizon)
    X_val, y_val = make_windows(val_s, input_len=input_len, horizon=horizon)
    X_test, y_test = make_windows(test_s, input_len=input_len, horizon=horizon)

    return PreparedData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
    )


def audit_no_scaler_leakage(
    scaler: TrainOnlyScaler,
    *,
    train: np.ndarray,
    full: np.ndarray,
    rtol: float = 1e-9,
) -> None:
    """Raise ``LeakageError`` if ``scaler`` wasn't fit on the training data alone.

    Compares the scaler's stored statistics against a fresh train-only fit. A
    scaler accidentally fit on the full series (the classic leak) will differ
    and trip this guard. ``full`` documents the leak we're guarding against and
    makes the call site self-explanatory.
    """
    expected = TrainOnlyScaler().fit(train)
    if not (
        math.isclose(scaler.mean_, expected.mean_, rel_tol=rtol, abs_tol=rtol)
        and math.isclose(scaler.std_, expected.std_, rel_tol=rtol, abs_tol=rtol)
    ):
        raise LeakageError(
            "scaler statistics do not match a train-only fit "
            f"(scaler mean/std = {scaler.mean_:.6g}/{scaler.std_:.6g}, "
            f"train-only = {expected.mean_:.6g}/{expected.std_:.6g}, "
            f"full-series mean = {float(np.asarray(full).mean()):.6g}). "
            "Was the scaler fit on more than the training segment?"
        )
