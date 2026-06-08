"""Train-only standardizer (z-score).

The second leakage guard. ``fit`` derives mean/std from the training data
*only*; ``transform`` applies those fixed statistics to any later data. Fitting
on the full series — a classic, silent leak — is structurally impossible here:
once fitted, transforming future data never touches the stored parameters.
"""

from __future__ import annotations

import numpy as np


class TrainOnlyScaler:
    """Standardize to zero mean / unit variance using train statistics only."""

    def __init__(self) -> None:
        self.mean_: float | None = None
        self.std_: float | None = None

    def fit(self, train: np.ndarray) -> TrainOnlyScaler:
        """Compute and store mean/std from ``train`` alone. Returns self."""
        train = np.asarray(train, dtype=float)
        self.mean_ = float(train.mean())
        std = float(train.std())
        # Guard against a constant training series collapsing to divide-by-zero.
        self.std_ = std if std > 0 else 1.0
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Standardize ``data`` with the fitted train statistics."""
        self._check_fitted()
        return (np.asarray(data, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, train: np.ndarray) -> np.ndarray:
        return self.fit(train).transform(train)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Map standardized values back to the original scale."""
        self._check_fitted()
        return np.asarray(data, dtype=float) * self.std_ + self.mean_

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("TrainOnlyScaler must be fit before transform")
