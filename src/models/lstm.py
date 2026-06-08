"""Stacked LSTM for direct multi-horizon forecasting.

Design choice (recorded in DECISIONS.md): **direct** multi-horizon output — the
model emits all ``horizon`` steps at once from the input window, rather than
forecasting one step and feeding it back (recursive). Direct avoids compounding
its own errors during training and matches the direct-target windowing in
``src/data/windowing.py``; the trade is more output parameters. The whiteboard
defense (Q5) is: recursive compounds, direct pays in parameters — I chose direct.

The model is just a participant: it is scored through the same walk-forward
harness as the baselines and only keeps its place if it beats them honestly.
"""

from __future__ import annotations

import torch
from torch import nn


class StackedLSTM(nn.Module):
    """A stacked LSTM whose last hidden state maps to a horizon-length forecast."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        horizon: int,
        input_size: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        # Inter-layer dropout only exists between stacked layers; PyTorch ignores
        # (and warns about) a dropout set on a single-layer LSTM, so zero it out.
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, seq_len[, input_size])`` to ``(batch, horizon)``."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len) -> single feature
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]  # final-timestep hidden state
        return self.head(last_step)
