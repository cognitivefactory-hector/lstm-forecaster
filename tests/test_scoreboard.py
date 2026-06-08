"""Tests for the per-horizon scoreboard — M2 acceptance ("prints per horizon").

The scoreboard is where honesty becomes visible: every model, every horizon,
MASE alongside RMSE/MAE — no single flattering aggregate.
"""

import numpy as np

from src.eval.scoreboard import build_scoreboard, format_scoreboard


def _fixture():
    # Ramp train -> MASE scale = 1.0. Two horizons.
    train = np.arange(50, dtype=float)
    y_true = np.array([[50.0, 51.0], [51.0, 52.0]])
    preds = {
        "naive": np.array([[49.0, 50.0], [50.0, 51.0]]),  # off by 1 each step
        "perfect": y_true.copy(),
    }
    return train, y_true, preds


def test_one_row_per_model_and_horizon():
    train, y_true, preds = _fixture()
    board = build_scoreboard(y_true, preds, train, season=1)
    assert len(board) == 2 * 2  # 2 models x 2 horizons


def test_has_a_row_for_every_horizon_step():
    train, y_true, preds = _fixture()
    board = build_scoreboard(y_true, preds, train, season=1)
    assert sorted(board["horizon"].unique()) == [1, 2]


def test_naive_scores_about_mase_one_and_perfect_scores_zero():
    train, y_true, preds = _fixture()
    board = build_scoreboard(y_true, preds, train, season=1)
    naive = board[board["model"] == "naive"]
    perfect = board[board["model"] == "perfect"]
    assert np.allclose(naive["mase"], 1.0)
    assert np.allclose(perfect["mase"], 0.0)
    assert np.allclose(perfect["rmse"], 0.0)


def test_format_scoreboard_is_a_readable_string():
    train, y_true, preds = _fixture()
    board = build_scoreboard(y_true, preds, train, season=1)
    text = format_scoreboard(board)
    assert isinstance(text, str)
    assert "naive" in text
    assert "MASE" in text.upper()
