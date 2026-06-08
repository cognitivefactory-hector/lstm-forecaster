"""ARIMA baseline with AIC-based order selection.

The second mandatory comparator. Order is chosen by minimizing AIC over a small,
documented grid (SPEC.md §7) — no sprawling hyperparameter search. statsmodels'
fitting can emit convergence/specification warnings on some orders; those are
suppressed locally so the scoreboard output stays clean, while genuine failures
to fit are scored as infinite AIC and simply lose the grid.
"""

from __future__ import annotations

import warnings

import numpy as np

Order = tuple[int, int, int]

# A small default grid: low-order AR/MA with optional single differencing.
DEFAULT_GRID: list[Order] = [
    (0, 0, 0),
    (1, 0, 0),
    (2, 0, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
]


def _fit(train: np.ndarray, order: Order):
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(train, order=order).fit()


def select_order(train: np.ndarray, candidate_orders: list[Order]) -> Order:
    """Return the order minimizing AIC over ``candidate_orders``.

    Orders that fail to fit are skipped (treated as +inf AIC).

    Raises:
        ValueError: if the grid is empty or no candidate fits.
    """
    if not candidate_orders:
        raise ValueError("candidate_orders is empty")

    train = np.asarray(train, dtype=float)
    best_order: Order | None = None
    best_aic = np.inf
    for order in candidate_orders:
        try:
            aic = _fit(train, order).aic
        except (ValueError, np.linalg.LinAlgError):
            continue
        if aic < best_aic:
            best_aic, best_order = aic, order

    if best_order is None:
        raise ValueError("no candidate order could be fit to the training series")
    return best_order


def arima_forecast(
    train: np.ndarray,
    horizon: int,
    *,
    order: Order | None = None,
    candidate_orders: list[Order] | None = None,
) -> tuple[np.ndarray, Order]:
    """Fit ARIMA and forecast ``horizon`` steps ahead.

    Provide an explicit ``order`` to skip selection, or a ``candidate_orders``
    grid to choose by AIC (defaults to :data:`DEFAULT_GRID`).

    Returns:
        ``(forecast, order)`` — the horizon-length forecast and the order used.
    """
    train = np.asarray(train, dtype=float)
    if order is None:
        order = select_order(train, candidate_orders or DEFAULT_GRID)

    fitted = _fit(train, order)
    forecast = np.asarray(fitted.forecast(steps=horizon), dtype=float)
    return forecast, order
