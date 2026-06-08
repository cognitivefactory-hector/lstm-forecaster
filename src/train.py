"""Reproducible LSTM training, checkpointing, and the walk-forward adapter.

Training is seeded end to end, scaling is fit on the training data only, and the
validation fold is the time-ordered tail of the training region — never a random
split. The model is saved with the scaler and config so inference reproduces the
training-time input distribution. ``make_lstm_forecaster`` wraps a trained model
as a ``forecaster(train, horizon)`` so the LSTM runs through the exact same
walk-forward backtest as the baselines.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data.scaling import TrainOnlyScaler
from src.data.splits import time_split
from src.data.windowing import make_windows
from src.models.lstm import StackedLSTM


def seed_everything(seed: int) -> None:
    """Seed Python/NumPy/Torch RNGs for reproducible training."""
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainResult:
    model: StackedLSTM
    scaler: TrainOnlyScaler
    config: dict
    history: dict = field(default_factory=lambda: {"train_loss": [], "val_loss": []})
    best_val_loss: float = float("inf")


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(x, dtype=np.float32))


def train_lstm(
    series: np.ndarray,
    *,
    input_len: int,
    horizon: int,
    hidden_size: int = 32,
    num_layers: int = 2,
    dropout: float = 0.1,
    val_frac: float = 0.2,
    epochs: int = 100,
    lr: float = 1e-2,
    seed: int = 0,
    patience: int = 10,
) -> TrainResult:
    """Train a stacked LSTM on ``series`` with train-only scaling.

    The validation fold (``val_frac`` of the series, the most recent slice) is
    used for early stopping; the best-val weights are restored at the end. Pass
    ``val_frac=0.0`` to disable validation and train on everything (used by the
    overfit sanity check).
    """
    seed_everything(seed)
    series = np.asarray(series, dtype=float).ravel()

    if val_frac > 0:
        train_raw, val_raw, _ = time_split(series, val_frac=val_frac, test_frac=0.0)
    else:
        train_raw, val_raw = series, np.empty(0)

    scaler = TrainOnlyScaler().fit(train_raw)
    X_train, y_train = make_windows(
        scaler.transform(train_raw), input_len=input_len, horizon=horizon
    )
    Xt, yt = _to_tensor(X_train).unsqueeze(-1), _to_tensor(y_train)

    has_val = len(val_raw) > input_len + horizon
    if has_val:
        X_val, y_val = make_windows(
            scaler.transform(val_raw), input_len=input_len, horizon=horizon
        )
        Xv, yv = _to_tensor(X_val).unsqueeze(-1), _to_tensor(y_val)

    model = StackedLSTM(
        hidden_size=hidden_size,
        num_layers=num_layers,
        horizon=horizon,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    result = TrainResult(model=model, scaler=scaler, config={
        "input_len": input_len,
        "horizon": horizon,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "seed": seed,
    })

    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_fn(model(Xt), yt)
        train_loss.backward()
        optimizer.step()
        result.history["train_loss"].append(float(train_loss.item()))

        if has_val:
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(Xv), yv).item())
            result.history["val_loss"].append(val_loss)

            if val_loss < result.best_val_loss:
                result.best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

    if has_val:
        model.load_state_dict(best_state)  # restore best-val weights
    else:
        result.best_val_loss = result.history["train_loss"][-1]
    return result


def make_lstm_forecaster(
    model: StackedLSTM,
    scaler: TrainOnlyScaler,
    input_len: int,
) -> Callable[[np.ndarray, int], np.ndarray]:
    """Wrap a trained model as a leak-free ``forecaster(train, horizon)``.

    Only the last ``input_len`` observations of the supplied past series feed the
    model; scaling uses the (train-only) fitted ``scaler``, and the forecast is
    returned on the original scale.
    """

    def forecaster(train: np.ndarray, horizon: int) -> np.ndarray:
        window = scaler.transform(np.asarray(train, dtype=float).ravel()[-input_len:])
        x = _to_tensor(window).reshape(1, input_len, 1)
        model.eval()
        with torch.no_grad():
            scaled_forecast = model(x).numpy().ravel()
        return scaler.inverse_transform(scaled_forecast)

    return forecaster


def save_checkpoint(result: TrainResult, path: str | Path) -> None:
    """Save model weights, scaler stats, and config to ``path`` (a .pt file)."""
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "scaler": {"mean": result.scaler.mean_, "std": result.scaler.std_},
            "config": result.config,
            "best_val_loss": result.best_val_loss,
        },
        path,
    )


def load_checkpoint(path: str | Path) -> tuple[StackedLSTM, TrainOnlyScaler, dict]:
    """Load a checkpoint into a fresh model + scaler. Returns (model, scaler, config)."""
    blob = torch.load(path, weights_only=False)
    config = blob["config"]
    model = StackedLSTM(
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        horizon=config["horizon"],
        dropout=config["dropout"],
    )
    model.load_state_dict(blob["state_dict"])

    scaler = TrainOnlyScaler()
    scaler.mean_ = blob["scaler"]["mean"]
    scaler.std_ = blob["scaler"]["std"]
    return model, scaler, config


def save_metrics(result: TrainResult, path: str | Path) -> None:
    """Write the training history + config + best val loss as JSON."""
    Path(path).write_text(
        json.dumps(
            {
                "config": result.config,
                "history": result.history,
                "best_val_loss": result.best_val_loss,
            },
            indent=2,
        )
    )


def run_training(series: np.ndarray, *, out_dir: str | Path, **kwargs) -> TrainResult:
    """Train and persist the checkpoint + metrics JSON under ``out_dir``.

    Thin orchestration over :func:`train_lstm`, :func:`save_checkpoint`, and
    :func:`save_metrics` — the reproducible entrypoint (SPEC.md §6).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = train_lstm(series, **kwargs)
    save_checkpoint(result, out_dir / "model.pt")
    save_metrics(result, out_dir / "metrics.json")
    return result


def main() -> None:
    """Train on the seeded synthetic series and write artifacts to checkpoints/."""
    from src.data.synthetic import make_seasonal_demand

    series = make_seasonal_demand(n=400, period=7, seed=0)
    result = run_training(
        series,
        out_dir="checkpoints",
        input_len=28,
        horizon=7,
        hidden_size=48,
        num_layers=2,
        dropout=0.1,
        val_frac=0.2,
        epochs=300,
        lr=1e-2,
        seed=0,
        patience=20,
    )
    print(
        f"Trained {len(result.history['train_loss'])} epochs; "
        f"best val loss {result.best_val_loss:.4f}. "
        "Saved checkpoints/model.pt + checkpoints/metrics.json"
    )


if __name__ == "__main__":
    main()
