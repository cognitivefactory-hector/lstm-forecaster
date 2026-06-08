"""Tests for the app's end-to-end evaluation pipeline — M6.

This is the demo's engine: train the LSTM, backtest it and the baselines through
the same walk-forward harness, and assemble the board + verdict + plots. The
Gradio layer (app.py) is just glue over this.
"""

from matplotlib.figure import Figure

from app.pipeline import run_evaluation
from src.data.synthetic import make_seasonal_demand

TINY = dict(
    horizon=3,
    input_len=10,
    hidden_size=16,
    num_layers=1,
    epochs=15,
    lr=0.01,
    seed=0,
)


def test_returns_all_demo_artifacts():
    series = make_seasonal_demand(n=140, period=7, seed=1)
    result = run_evaluation(series, season=7, **TINY)

    assert isinstance(result.forecast_fig, Figure)
    assert isinstance(result.error_fig, Figure)
    assert "VERDICT" in result.summary


def test_board_contains_the_lstm_and_both_baselines():
    series = make_seasonal_demand(n=140, period=7, seed=1)
    result = run_evaluation(series, season=7, **TINY)
    models = set(result.board["model"].unique())
    assert {"naive", "seasonal_naive", "lstm"} <= models


def test_verdict_compares_lstm_against_the_seasonal_baseline():
    series = make_seasonal_demand(n=140, period=7, seed=1)
    result = run_evaluation(series, season=7, **TINY)
    assert result.verdict.model == "lstm"
    assert result.verdict.baseline == "seasonal_naive"
    assert result.verdict.overall in {"wins", "ties", "loses"}


def test_is_reproducible_for_a_fixed_seed():
    series = make_seasonal_demand(n=140, period=7, seed=1)
    a = run_evaluation(series, season=7, **TINY)
    b = run_evaluation(series, season=7, **TINY)
    assert a.summary == b.summary
