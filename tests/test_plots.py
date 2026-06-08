"""Tests for the M5 plots: forecast-vs-actual (baseline overlaid) and error-by-horizon.

We assert structure, not pixels: the baseline must always be drawn alongside the
model, and error-by-horizon must show one line per model so degradation across
the horizon is visible.
"""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.eval.plots import plot_error_by_horizon, plot_forecast_vs_actual


def _board():
    rows = []
    for model, vals in {"lstm": [1.2, 1.3], "seasonal_naive": [1.0, 1.1]}.items():
        for h, v in enumerate(vals, start=1):
            rows.append({"model": model, "horizon": h, "mase": v, "rmse": v, "mae": v})
    return pd.DataFrame(rows)


def test_forecast_plot_draws_actual_plus_every_model():
    actual = np.array([10.0, 11.0, 12.0])
    forecasts = {
        "lstm": np.array([10.5, 11.2, 11.8]),
        "seasonal_naive": np.array([9.8, 10.9, 12.3]),
    }
    fig = plot_forecast_vs_actual(actual, forecasts)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert len(ax.get_lines()) == 3  # actual + 2 models

    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "actual" in labels
    assert "seasonal_naive" in labels  # baseline always overlaid


def test_error_by_horizon_draws_one_line_per_model():
    fig = plot_error_by_horizon(_board(), metric="mase")
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    # One labeled line per model (the reference line is unlabeled).
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert labels == {"lstm", "seasonal_naive"}


def test_error_by_horizon_draws_the_beat_naive_reference_line_for_mase():
    fig = plot_error_by_horizon(_board(), metric="mase")
    ax = fig.axes[0]
    # A dashed horizontal reference at MASE = 1.0 (the threshold to beat naive).
    flat_at_one = [
        ln
        for ln in ax.get_lines()
        if np.allclose(ln.get_ydata(), 1.0) and ln.get_linestyle() == "--"
    ]
    assert len(flat_at_one) == 1


def test_error_by_horizon_labels_the_metric_axis():
    fig = plot_error_by_horizon(_board(), metric="mase")
    ax = fig.axes[0]
    assert "MASE" in ax.get_ylabel().upper()
    assert "horizon" in ax.get_xlabel().lower()
