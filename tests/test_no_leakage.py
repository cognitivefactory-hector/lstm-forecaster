"""The crown-jewel leakage tests (M1).

If any of these can be made to pass while the evaluation secretly peeks at the
future, the whole project's thesis is void. They assert the three guards
end-to-end and prove the audit fails *loudly* on an accidental full-series fit.
"""

import numpy as np
import pytest

from src.data.harness import LeakageError, audit_no_scaler_leakage, prepare_splits
from src.data.scaling import TrainOnlyScaler
from src.data.synthetic import make_seasonal_demand

# A trending series: train statistics differ from full-series statistics, so a
# full-series scaler fit is detectably different from a train-only fit.
SERIES = make_seasonal_demand(n=400, period=7, seed=7, trend=0.2)
PREP_KW = dict(input_len=20, horizon=5, val_frac=0.2, test_frac=0.2)


def test_scaler_parameters_depend_only_on_training_data():
    prep = prepare_splits(SERIES, **PREP_KW)
    n_test = round(len(SERIES) * PREP_KW["test_frac"])
    n_val = round(len(SERIES) * PREP_KW["val_frac"])
    train_raw = SERIES[: len(SERIES) - n_val - n_test]

    assert prep.scaler.mean_ == pytest.approx(float(train_raw.mean()))
    # ...and crucially NOT the full-series statistics.
    assert prep.scaler.mean_ != pytest.approx(float(SERIES.mean()))


def test_no_window_crosses_the_forecast_origin():
    # Test-set input windows are drawn entirely from the post-origin segment:
    # inverse-transforming a test window recovers raw values from the test tail,
    # never from train. And the train window count matches the train segment
    # alone — so no window bridges a split boundary.
    prep = prepare_splits(SERIES, **PREP_KW)
    n_test = round(len(SERIES) * PREP_KW["test_frac"])
    n_val = round(len(SERIES) * PREP_KW["val_frac"])
    n_train = len(SERIES) - n_val - n_test
    train_raw, test_raw = SERIES[:n_train], SERIES[n_train + n_val :]

    expected_train_windows = n_train - PREP_KW["input_len"] - PREP_KW["horizon"] + 1
    assert len(prep.X_train) == expected_train_windows

    recovered_test_input = prep.scaler.inverse_transform(prep.X_test[0])
    np.testing.assert_allclose(recovered_test_input, test_raw[: PREP_KW["input_len"]])
    recovered_train_input = prep.scaler.inverse_transform(prep.X_train[0])
    np.testing.assert_allclose(recovered_train_input, train_raw[: PREP_KW["input_len"]])


def test_windows_preserve_time_order_no_shuffling():
    prep = prepare_splits(SERIES, **PREP_KW)
    # Each window is the previous one slid forward by exactly one step: the tail
    # of window i equals the head of window i+1. Shuffling rows would break this.
    np.testing.assert_allclose(prep.X_train[:-1, 1:], prep.X_train[1:, :-1])


def test_audit_passes_for_a_correct_train_only_fit():
    train_raw = SERIES[:200]
    scaler = TrainOnlyScaler().fit(train_raw)
    # Should not raise.
    audit_no_scaler_leakage(scaler, train=train_raw, full=SERIES)


def test_audit_fails_loudly_when_scaler_is_fit_on_the_full_series():
    train_raw = SERIES[:200]
    leaky = TrainOnlyScaler().fit(SERIES)  # the classic leak
    with pytest.raises(LeakageError):
        audit_no_scaler_leakage(leaky, train=train_raw, full=SERIES)
