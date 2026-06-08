"""Tests for the plain-language verdict — M5, the honesty centerpiece.

The verdict states, per horizon and overall, whether the deep model beats the
baseline — and names the horizons where it LOSES. A portfolio that only reports
wins carries no signal; this module makes the losses first-class.
"""

import pandas as pd
import pytest

from src.eval.verdict import build_verdict


def _board(model_mase, baseline_mase, model="lstm", baseline="seasonal_naive"):
    rows = []
    for h, m in enumerate(model_mase, start=1):
        rows.append({"model": model, "horizon": h, "mase": m, "rmse": m, "mae": m})
    for h, b in enumerate(baseline_mase, start=1):
        rows.append({"model": baseline, "horizon": h, "mase": b, "rmse": b, "mae": b})
    return pd.DataFrame(rows)


def test_model_winning_every_horizon_reads_as_a_win():
    board = _board([0.5, 0.6, 0.7], [1.0, 1.0, 1.0])
    v = build_verdict(board, model="lstm", baseline="seasonal_naive")
    assert v.overall == "wins"
    assert v.wins == [1, 2, 3]
    assert v.losses == []


def test_model_losing_overall_reads_as_a_loss_and_names_the_horizons():
    board = _board([1.2, 1.3, 1.1], [1.0, 1.0, 1.0])
    v = build_verdict(board, model="lstm", baseline="seasonal_naive")
    assert v.overall == "loses"
    assert v.losses == [1, 2, 3]
    assert "LOSES" in v.text.upper()


def test_mixed_result_records_both_wins_and_losses():
    # Wins at h1, loses at h2, ties at h3 (within 5%).
    board = _board([0.5, 1.5, 1.02], [1.0, 1.0, 1.0])
    v = build_verdict(board, model="lstm", baseline="seasonal_naive", tie_rel=0.05)
    assert v.wins == [1]
    assert v.losses == [2]
    assert v.ties == [3]


def test_near_equal_means_overall_tie():
    board = _board([1.01, 0.99, 1.0], [1.0, 1.0, 1.0])
    v = build_verdict(board, model="lstm", baseline="seasonal_naive", tie_rel=0.05)
    assert v.overall == "ties"


def test_text_names_both_models_and_reports_mean_mase():
    board = _board([1.2, 1.3, 1.1], [1.0, 1.0, 1.0])
    v = build_verdict(board, model="lstm", baseline="seasonal_naive")
    assert "lstm" in v.text
    assert "seasonal_naive" in v.text
    assert v.model_mean_mase == pytest.approx(1.2)
    assert v.baseline_mean_mase == pytest.approx(1.0)


def test_raises_when_a_model_is_missing_from_the_board():
    board = _board([1.0], [1.0])
    with pytest.raises(ValueError):
        build_verdict(board, model="transformer", baseline="seasonal_naive")
