"""Tests for reproducible LSTM training + the walk-forward adapter — M4.

The model gets sanity tests, not accuracy assertions (PLAN testing strategy):
deterministic training under a seed, an overfit-a-tiny-set check that proves the
loop actually learns, a checkpoint round-trip, and the leak-free forecaster
adapter that lets the LSTM run through the same backtest as the baselines.
"""

import numpy as np
import torch

from src.data.synthetic import make_seasonal_demand
from src.eval.walk_forward import walk_forward
from src.train import (
    load_checkpoint,
    make_lstm_forecaster,
    save_checkpoint,
    train_lstm,
)

TINY = dict(input_len=10, horizon=3, hidden_size=16, num_layers=1)


def test_training_reduces_the_loss():
    series = make_seasonal_demand(n=200, period=7, seed=0)
    result = train_lstm(series, **TINY, val_frac=0.2, epochs=50, lr=0.01, seed=0)
    assert result.history["train_loss"][-1] < result.history["train_loss"][0]


def test_training_is_reproducible_under_a_seed():
    series = make_seasonal_demand(n=200, period=7, seed=0)
    a = train_lstm(series, **TINY, val_frac=0.2, epochs=30, lr=0.01, seed=7)
    b = train_lstm(series, **TINY, val_frac=0.2, epochs=30, lr=0.01, seed=7)
    assert a.history["train_loss"] == b.history["train_loss"]


def test_overfits_a_tiny_set_to_near_zero_loss():
    # The canonical sanity check: with no validation split and enough epochs, the
    # loop must be able to drive train loss to ~0 on a handful of windows. If it
    # can't, the training loop itself is broken.
    series = make_seasonal_demand(n=40, period=7, seed=1)
    result = train_lstm(
        series,
        input_len=10,
        horizon=3,
        hidden_size=32,
        num_layers=1,
        val_frac=0.0,
        epochs=600,
        lr=0.01,
        seed=0,
    )
    assert result.history["train_loss"][-1] < 0.02


def test_early_stopping_keeps_history_within_epoch_budget():
    series = make_seasonal_demand(n=200, period=7, seed=2)
    result = train_lstm(
        series, **TINY, val_frac=0.2, epochs=100, lr=0.01, seed=0, patience=5
    )
    assert 1 <= len(result.history["train_loss"]) <= 100
    assert len(result.history["val_loss"]) == len(result.history["train_loss"])


def test_checkpoint_round_trips(tmp_path):
    series = make_seasonal_demand(n=200, period=7, seed=0)
    result = train_lstm(series, **TINY, val_frac=0.2, epochs=20, lr=0.01, seed=0)
    path = tmp_path / "model.pt"
    save_checkpoint(result, path)

    model, scaler, config = load_checkpoint(path)
    x = torch.randn(2, TINY["input_len"], 1)
    result.model.eval()
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x), result.model(x))
    assert config["input_len"] == TINY["input_len"]
    assert scaler.mean_ == result.scaler.mean_


def test_forecaster_adapter_produces_horizon_length_forecasts():
    series = make_seasonal_demand(n=200, period=7, seed=0)
    result = train_lstm(series, **TINY, val_frac=0.2, epochs=20, lr=0.01, seed=0)
    forecaster = make_lstm_forecaster(result.model, result.scaler, input_len=10)
    out = forecaster(series[:150], horizon=3)
    assert out.shape == (3,)


def test_forecaster_only_uses_the_last_input_window():
    # Leak-free + history-agnostic: only the final input_len points of whatever
    # past series it's given can affect the forecast. Changing earlier history
    # must not change the output.
    series = make_seasonal_demand(n=200, period=7, seed=0)
    result = train_lstm(series, **TINY, val_frac=0.2, epochs=20, lr=0.01, seed=0)
    forecaster = make_lstm_forecaster(result.model, result.scaler, input_len=10)

    base = forecaster(series[:150], horizon=3)
    altered = series[:150].copy()
    altered[:100] += 999.0  # mangle the distant past, keep the last 10 intact
    np.testing.assert_allclose(forecaster(altered, horizon=3), base)


def test_lstm_runs_through_the_walk_forward_harness():
    # The acceptance: the LSTM is evaluated through the SAME engine as baselines.
    series = make_seasonal_demand(n=200, period=7, seed=0)
    result = train_lstm(series, **TINY, val_frac=0.2, epochs=20, lr=0.01, seed=0)
    forecaster = make_lstm_forecaster(result.model, result.scaler, input_len=10)

    backtest = walk_forward(
        series, forecaster, horizon=3, initial_train=150, step=5
    )
    assert backtest.y_pred.shape[1] == 3
    assert len(backtest.folds) >= 1
