"""Per-horizon scoreboard — the honest comparison table.

Tabulates MASE / RMSE / MAE for every model at every horizon step. Reporting
per horizon (never a single aggregate) keeps the compounding of multistep error
visible, and showing every model side by side keeps the baseline always in
frame — including any series/horizon where a baseline beats the deep model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.metrics import mae, mase, per_horizon, rmse


def build_scoreboard(
    y_true: np.ndarray,
    model_preds: dict[str, np.ndarray],
    y_train: np.ndarray,
    *,
    season: int = 1,
) -> pd.DataFrame:
    """Build a tidy per-(model, horizon) scoreboard.

    Args:
        y_true: Actuals shaped ``(n_windows, horizon)``.
        model_preds: Mapping of model name -> predictions of the same shape.
        y_train: Training series used for the MASE scale.
        season: Seasonal lag for the MASE baseline.

    Returns:
        A long-form DataFrame with columns
        ``model, horizon, mase, rmse, mae`` (horizon is 1-indexed).
    """
    y_true = np.asarray(y_true, dtype=float)
    horizon = y_true.shape[1]

    rows = []
    for name, preds in model_preds.items():
        preds = np.asarray(preds, dtype=float)
        per_mase = per_horizon(mase, y_true, preds, y_train=y_train, season=season)
        per_rmse = per_horizon(rmse, y_true, preds)
        per_mae = per_horizon(mae, y_true, preds)
        for h in range(horizon):
            rows.append(
                {
                    "model": name,
                    "horizon": h + 1,
                    "mase": per_mase[h],
                    "rmse": per_rmse[h],
                    "mae": per_mae[h],
                }
            )
    return pd.DataFrame(rows, columns=["model", "horizon", "mase", "rmse", "mae"])


def format_scoreboard(board: pd.DataFrame) -> str:
    """Render the scoreboard as a readable, fixed-width table for printing."""
    display = board.rename(
        columns={
            "model": "Model",
            "horizon": "Horizon",
            "mase": "MASE",
            "rmse": "RMSE",
            "mae": "MAE",
        }
    )
    return display.to_string(
        index=False,
        formatters={
            "MASE": "{:.3f}".format,
            "RMSE": "{:.3f}".format,
            "MAE": "{:.3f}".format,
        },
    )
