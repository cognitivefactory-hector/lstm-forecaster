"""Tests for the reproducible training entrypoint (saves artifacts) — M4."""

import json

from src.data.synthetic import make_seasonal_demand
from src.train import load_checkpoint, run_training


def test_run_training_writes_checkpoint_and_metrics(tmp_path):
    series = make_seasonal_demand(n=200, period=7, seed=0)
    result = run_training(
        series,
        out_dir=tmp_path,
        input_len=10,
        horizon=3,
        hidden_size=16,
        num_layers=1,
        val_frac=0.2,
        epochs=20,
        lr=0.01,
        seed=0,
    )

    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "metrics.json").exists()

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["config"]["horizon"] == 3
    assert len(metrics["history"]["train_loss"]) >= 1

    # The saved checkpoint reloads into a working model.
    model, scaler, config = load_checkpoint(tmp_path / "model.pt")
    assert config["input_len"] == 10
    assert scaler.mean_ == result.scaler.mean_
