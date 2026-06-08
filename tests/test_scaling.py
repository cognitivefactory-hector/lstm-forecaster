"""Tests for the train-only standardizer (second leakage guard) — M1."""

import numpy as np
import pytest

from src.data.scaling import TrainOnlyScaler


def test_fit_uses_only_the_data_it_was_given():
    train = np.array([0.0, 2.0, 4.0, 6.0, 8.0])  # mean 4, std 2*sqrt(2)
    scaler = TrainOnlyScaler().fit(train)
    assert scaler.mean_ == pytest.approx(4.0)
    assert scaler.std_ == pytest.approx(train.std())


def test_transform_standardizes_with_train_statistics():
    train = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    scaler = TrainOnlyScaler().fit(train)
    out = scaler.transform(train)
    assert out.mean() == pytest.approx(0.0, abs=1e-9)
    assert out.std() == pytest.approx(1.0)


def test_future_data_does_not_change_fitted_parameters():
    # The crux of the no-leakage guard: stats come only from train. Transforming
    # later (larger) data must not retroactively alter the fitted parameters.
    train = np.array([1.0, 2.0, 3.0, 4.0])
    scaler = TrainOnlyScaler().fit(train)
    mean_before, std_before = scaler.mean_, scaler.std_

    future = np.array([100.0, 200.0, 300.0])
    scaler.transform(future)

    assert scaler.mean_ == mean_before
    assert scaler.std_ == std_before


def test_inverse_transform_round_trips():
    train = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    scaler = TrainOnlyScaler().fit(train)
    data = np.array([15.0, 35.0, 55.0])
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(data)), data)


def test_transform_before_fit_raises():
    with pytest.raises(ValueError):
        TrainOnlyScaler().transform(np.array([1.0, 2.0]))
