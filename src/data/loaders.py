"""Public series loader (keyless, via yfinance).

Returns a clean, time-ordered, 1-D float array — the contract the leak-free
harness expects. Network access is injectable (``_download``) so the parsing
logic is tested offline; the default path lazily imports ``yfinance`` so the
dependency is only needed when actually fetching.

Provenance/limits are the caller's to document in the README (SPEC.md §5). The
financial variant forecasts method, not market-beating claims — not advice.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def _yf_download(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf  # lazy: only needed for real network fetches

    return yf.download(ticker, start=start, end=end, progress=False)


def load_yfinance_series(
    ticker: str,
    start: str,
    end: str,
    *,
    column: str = "Close",
    _download: Callable[..., pd.DataFrame] | None = None,
) -> np.ndarray:
    """Fetch one column of a ticker's history as a clean, ordered 1-D series.

    Args:
        ticker: Symbol to fetch (e.g. ``"AAPL"``).
        start: Inclusive start date (``YYYY-MM-DD``).
        end: Exclusive end date (``YYYY-MM-DD``).
        column: Which price column to extract.
        _download: Test/seam override for the network fetch.

    Returns:
        A 1-D float array sorted by date with missing values dropped.

    Raises:
        ValueError: if the download is empty.
    """
    download = _download or _yf_download
    frame = download(ticker, start, end)
    if len(frame) == 0:
        raise ValueError(f"no data returned for {ticker!r} in [{start}, {end})")

    series = frame[column].sort_index().dropna()
    return series.to_numpy(dtype=float).ravel()
