"""Tests for the public (yfinance) series loader — M1.

Network I/O is injected so these run offline and deterministically. We test the
contract the harness depends on: a clean, time-ordered, 1-D float series.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.loaders import load_yfinance_series


def _fake_frame():
    # Out of chronological order on purpose, with a gap (NaN) to be dropped.
    idx = pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02", "2023-01-04"])
    return pd.DataFrame({"Close": [12.0, 10.0, np.nan, 13.0]}, index=idx)


def test_returns_1d_float_series():
    series = load_yfinance_series(
        "TEST", "2023-01-01", "2023-01-05", _download=lambda *a, **k: _fake_frame()
    )
    assert series.ndim == 1
    assert series.dtype == float


def test_sorts_by_time_and_drops_missing_values():
    series = load_yfinance_series(
        "TEST", "2023-01-01", "2023-01-05", _download=lambda *a, **k: _fake_frame()
    )
    # Sorted by date, NaN row removed: 10.0 (01-01), 13.0 (01-04), 12.0 (01-03)?
    # After sorting by index: 01-01=10, 01-02=NaN(drop), 01-03=12, 01-04=13.
    np.testing.assert_array_equal(series, [10.0, 12.0, 13.0])


def test_raises_on_empty_download():
    empty = pd.DataFrame({"Close": []})
    with pytest.raises(ValueError):
        load_yfinance_series(
            "TEST", "2023-01-01", "2023-01-05", _download=lambda *a, **k: empty
        )
