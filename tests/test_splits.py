"""Tests for time-ordered splitting (no shuffling, contiguous, ordered) — M1."""

import numpy as np
import pytest

from src.data.splits import time_split


def test_split_sizes_follow_the_requested_fractions():
    series = np.arange(100, dtype=float)
    train, val, test = time_split(series, val_frac=0.2, test_frac=0.2)
    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20


def test_splits_are_contiguous_and_in_time_order():
    # train is the earliest segment, then val, then test — concatenating the
    # three reconstructs the original series exactly (no shuffle, no gaps).
    series = np.arange(50, dtype=float)
    train, val, test = time_split(series, val_frac=0.2, test_frac=0.2)
    np.testing.assert_array_equal(np.concatenate([train, val, test]), series)


def test_train_strictly_precedes_test_in_time():
    series = np.arange(50, dtype=float)
    train, _, test = time_split(series, val_frac=0.2, test_frac=0.2)
    assert train.max() < test.min()


def test_rejects_fractions_that_leave_no_training_data():
    series = np.arange(50, dtype=float)
    with pytest.raises(ValueError):
        time_split(series, val_frac=0.6, test_frac=0.6)
