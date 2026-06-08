"""Tests for the stacked LSTM module (direct multi-horizon) — M4.

The model is a *participant* measured through the same harness as the baselines,
not the hero. These tests pin its mechanics: correct output shape, determinism
under a seed (so results are reproducible), and inter-layer dropout wired only
where it's valid.
"""

import torch

from src.models.lstm import StackedLSTM


def test_forward_outputs_one_value_per_horizon_step():
    model = StackedLSTM(hidden_size=8, num_layers=2, horizon=5)
    x = torch.zeros(4, 12, 1)  # (batch, seq_len, input_size)
    out = model(x)
    assert out.shape == (4, 5)


def test_accepts_2d_input_by_treating_it_as_single_feature():
    model = StackedLSTM(hidden_size=8, num_layers=1, horizon=3)
    x = torch.zeros(4, 10)  # (batch, seq_len)
    assert model(x).shape == (4, 3)


def test_forward_is_deterministic_under_a_seed():
    x = torch.randn(3, 10, 1)
    torch.manual_seed(123)
    a = StackedLSTM(hidden_size=16, num_layers=2, horizon=4)(x)
    torch.manual_seed(123)
    b = StackedLSTM(hidden_size=16, num_layers=2, horizon=4)(x)
    torch.testing.assert_close(a, b)


def test_different_seeds_give_different_initial_weights():
    x = torch.randn(3, 10, 1)
    torch.manual_seed(1)
    a = StackedLSTM(hidden_size=16, num_layers=2, horizon=4)(x)
    torch.manual_seed(2)
    b = StackedLSTM(hidden_size=16, num_layers=2, horizon=4)(x)
    assert not torch.allclose(a, b)


def test_single_layer_disables_inter_layer_dropout():
    # nn.LSTM warns/errors if dropout is set with one layer; we must zero it out.
    model = StackedLSTM(hidden_size=8, num_layers=1, horizon=2, dropout=0.5)
    assert model.lstm.dropout == 0.0


def test_multi_layer_keeps_requested_dropout():
    model = StackedLSTM(hidden_size=8, num_layers=3, horizon=2, dropout=0.3)
    assert model.lstm.dropout == 0.3
