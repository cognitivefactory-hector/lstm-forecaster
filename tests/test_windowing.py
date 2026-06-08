"""Tests for the sliding-window builder (multistep targets) — M1 crown jewel.

Shapes and targets must be exactly right: a silent off-by-one here would let a
target step leak into the input window, which is leakage by another name.
"""

import numpy as np
import pytest

from src.data.windowing import make_windows


def test_shapes_for_multistep_targets():
    series = np.arange(20, dtype=float)
    input_len, horizon = 5, 3
    X, y = make_windows(series, input_len=input_len, horizon=horizon)

    n_expected = len(series) - input_len - horizon + 1  # 20 - 5 - 3 + 1 = 13
    assert X.shape == (n_expected, input_len)
    assert y.shape == (n_expected, horizon)


def test_first_window_holds_the_earliest_input_and_its_following_targets():
    series = np.arange(20, dtype=float)
    X, y = make_windows(series, input_len=5, horizon=3)

    np.testing.assert_array_equal(X[0], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(y[0], [5, 6, 7])  # the steps right after input


def test_targets_never_overlap_their_own_input_window():
    # For every window, the last input value precedes the first target value.
    series = np.arange(50, dtype=float)
    X, y = make_windows(series, input_len=8, horizon=4)
    assert np.all(X[:, -1] < y[:, 0])


def test_windows_are_in_time_order_not_shuffled():
    # Each successive window is the previous one shifted by exactly one step.
    series = np.arange(30, dtype=float)
    X, _ = make_windows(series, input_len=6, horizon=2)
    np.testing.assert_array_equal(X[1:, 0], X[:-1, 0] + 1)


def test_raises_when_series_too_short_for_a_single_window():
    series = np.arange(5, dtype=float)
    with pytest.raises(ValueError):
        make_windows(series, input_len=5, horizon=3)  # needs >= 8 points
