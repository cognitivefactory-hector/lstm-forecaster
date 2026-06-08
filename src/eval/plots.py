"""Plots for the honest scoreboard: forecast-vs-actual and error-by-horizon.

Figures are built directly (no pyplot global state), so this is headless-safe
and the returned ``Figure`` objects drop straight into the Gradio app (M6). Two
views, both in service of honesty: the baseline is always overlaid on the
forecast, and error is shown per horizon so multistep degradation can't hide.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def plot_forecast_vs_actual(
    actual: np.ndarray,
    forecasts: dict[str, np.ndarray],
    *,
    title: str = "Forecast vs. actual",
) -> Figure:
    """Plot the actual series with every model's forecast overlaid.

    Args:
        actual: The realized values over the forecast horizon.
        forecasts: Mapping of model name -> forecast (same length as ``actual``).
            The baseline belongs here too — it is always drawn alongside.
    """
    actual = np.asarray(actual, dtype=float).ravel()
    steps = np.arange(1, len(actual) + 1)

    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    ax.plot(steps, actual, label="actual", color="black", linewidth=2)
    for name, forecast in forecasts.items():
        ax.plot(steps, np.asarray(forecast, dtype=float).ravel(), marker="o", label=name)

    ax.set_title(title)
    ax.set_xlabel("horizon step")
    ax.set_ylabel("value")
    ax.set_xticks(steps)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_error_by_horizon(
    board: pd.DataFrame,
    *,
    metric: str = "mase",
    title: str | None = None,
) -> Figure:
    """Plot ``metric`` against the horizon, one line per model.

    Reads the per-(model, horizon) scoreboard so error growth across the horizon
    is explicit. A dashed reference line at 1.0 is drawn for MASE (the
    beat-the-naive threshold).
    """
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    for name, group in board.groupby("model"):
        group = group.sort_values("horizon")
        ax.plot(group["horizon"], group[metric], marker="o", label=name)

    if metric == "mase":
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)  # beat-naive line

    ax.set_title(title or f"{metric.upper()} by horizon")
    ax.set_xlabel("horizon step")
    ax.set_ylabel(metric.upper())
    ax.legend()
    fig.tight_layout()
    return fig
