"""End-to-end evaluation pipeline for the demo.

Trains the LSTM on the pre-origin region, backtests it and the baselines through
the *same* walk-forward harness, then assembles the per-horizon scoreboard, the
plain-language verdict, and the two plots. All the honesty machinery from M1-M5,
wired into one call the Gradio layer can drive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.eval.plots import plot_error_by_horizon, plot_forecast_vs_actual
from src.eval.scoreboard import build_scoreboard, format_scoreboard
from src.eval.verdict import Verdict, build_verdict
from src.eval.walk_forward import walk_forward
from src.models.baselines import naive_forecast, seasonal_naive_forecast
from src.train import make_lstm_forecaster, train_lstm


@dataclass
class EvaluationResult:
    board: pd.DataFrame
    verdict: Verdict
    forecast_fig: Figure
    error_fig: Figure
    summary: str


def run_evaluation(
    series: np.ndarray,
    *,
    season: int,
    horizon: int,
    input_len: int,
    initial_train_frac: float = 0.7,
    step: int | None = None,
    hidden_size: int = 48,
    num_layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 150,
    lr: float = 1e-2,
    seed: int = 0,
    patience: int = 10,
) -> EvaluationResult:
    """Train, backtest, and report the LSTM against the baselines.

    The LSTM is trained once on ``series[:origin]`` (leak-free) and, like the
    baselines, evaluated only on folds at or after that origin.
    """
    series = np.asarray(series, dtype=float).ravel()
    origin = int(len(series) * initial_train_frac)
    step = step or horizon

    trained = train_lstm(
        series[:origin],
        input_len=input_len,
        horizon=horizon,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        epochs=epochs,
        lr=lr,
        seed=seed,
        patience=patience,
    )
    lstm_forecaster = make_lstm_forecaster(trained.model, trained.scaler, input_len)

    forecasters = {
        "naive": lambda tr, h: naive_forecast(tr, h),
        "seasonal_naive": lambda tr, h: seasonal_naive_forecast(tr, h, period=season),
        "lstm": lstm_forecaster,
    }
    backtests = {
        name: walk_forward(series, fc, horizon=horizon, initial_train=origin, step=step)
        for name, fc in forecasters.items()
    }

    y_true = backtests["naive"].y_true
    preds = {name: bt.y_pred for name, bt in backtests.items()}
    board = build_scoreboard(y_true, preds, series[:origin], season=season)
    verdict = build_verdict(board, model="lstm", baseline="seasonal_naive")

    last_fold = {name: bt.folds[-1].forecast for name, bt in backtests.items()}
    forecast_fig = plot_forecast_vs_actual(
        backtests["naive"].folds[-1].actual,
        last_fold,
        title="Most recent fold: forecast vs. actual",
    )
    error_fig = plot_error_by_horizon(board, metric="mase")

    summary = f"{verdict.text}\n\nPer-horizon scoreboard:\n{format_scoreboard(board)}"
    return EvaluationResult(
        board=board,
        verdict=verdict,
        forecast_fig=forecast_fig,
        error_fig=error_fig,
        summary=summary,
    )
