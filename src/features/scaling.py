"""Standardization scaler for neural-net inputs.

Trees (XGBoost) need no scaling; LSTM/CNN-LSTM do. Because fit() computes
mean/std directly from whatever DataFrame/array it is given, this is the
first place in the pipeline that could silently leak validation or test
statistics into a model - per CLAUDE.md rule 3, scalers and feature
statistics are fit on TRAINING data only. So fit() may run exactly once
per instance: a second call raises RuntimeError rather than silently
refitting on a different split.

numpy/pandas only - no torch, so this can be unit tested without a GPU.
"""

import numpy as np
import pandas as pd

# Below this, a column is treated as constant: dividing by it would blow
# up to a huge or infinite scaled value rather than a division by exactly
# zero, so an exact-zero check alone would not catch it.
ZERO_VARIANCE_EPS = 1e-12


def _to_matrix(X):
    """Return (2D float ndarray, column labels) for a DataFrame, Series, or
    plain 1D/2D ndarray. Labels are the DataFrame's columns, the Series'
    name (a single label), or synthetic "col_i" for a bare ndarray - used
    only for error messages and to_dict, never to reorder or look up data.
    """
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float), list(X.columns)
    if isinstance(X, pd.Series):
        name = X.name if X.name is not None else "value"
        return X.to_numpy(dtype=float).reshape(-1, 1), [name]
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1), ["value"]
    if arr.ndim == 2:
        return arr, [f"col_{i}" for i in range(arr.shape[1])]
    raise ValueError(f"Scaler expects 1D or 2D input, got ndim={arr.ndim}")


def _like(X, arr2d):
    """Rewrap a transformed 2D ndarray to match the type/shape of the
    original input: DataFrame in -> DataFrame out (same index/columns),
    Series in -> Series out, 1D ndarray in -> 1D ndarray out, 2D ndarray
    in -> 2D ndarray out.
    """
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(arr2d, index=X.index, columns=X.columns)
    if isinstance(X, pd.Series):
        return pd.Series(arr2d[:, 0], index=X.index, name=X.name)
    arr = np.asarray(X)
    if arr.ndim == 1:
        return arr2d[:, 0]
    return arr2d


class Scaler:
    """Per-column standardization: (x - mean) / std.

    fit() must be called on TRAINING data ONLY (CLAUDE.md rule 3) and
    exactly once per instance - a second call raises RuntimeError, so a
    scaler can never be silently refit on validation or test statistics
    partway through an experiment. Raises ValueError if any column has
    (near) zero variance in the training data.

    Accepts a pandas DataFrame, a pandas Series, or a plain 1D/2D numpy
    array; transform/inverse_transform return whatever type they were
    given. Column count must match what fit() saw.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self._columns = None
        self._fitted = False

    def fit(self, X_train):
        if self._fitted:
            raise RuntimeError(
                "Scaler.fit called twice on the same instance. Per "
                "CLAUDE.md rule 3, a scaler is fit on training data once; "
                "refitting could silently pull in validation/test "
                "statistics. Create a new Scaler instead."
            )
        arr, columns = _to_matrix(X_train)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=0)

        zero_var = [columns[i] for i in np.where(std <= ZERO_VARIANCE_EPS)[0]]
        if zero_var:
            raise ValueError(
                f"Scaler.fit: column(s) {zero_var} have zero variance in "
                "the training data - cannot standardize a constant column"
            )

        self.mean_ = mean
        self.std_ = std
        self._columns = columns
        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Scaler is not fitted - call fit() first")

    def _check_shape(self, arr):
        if arr.shape[1] != len(self._columns):
            raise ValueError(
                f"Scaler: expected {len(self._columns)} column(s) "
                f"{self._columns}, got {arr.shape[1]}"
            )

    def transform(self, X):
        self._check_fitted()
        arr, _columns = _to_matrix(X)
        self._check_shape(arr)
        scaled = (arr - self.mean_) / self.std_
        return _like(X, scaled)

    def fit_transform(self, X_train):
        self.fit(X_train)
        return self.transform(X_train)

    def inverse_transform(self, X):
        """Invert a previous transform() - e.g. to map a scaled target
        prediction back to physical units.
        """
        self._check_fitted()
        arr, _columns = _to_matrix(X)
        self._check_shape(arr)
        original = arr * self.std_ + self.mean_
        return _like(X, original)

    def to_dict(self):
        """Fitted parameters as plain Python types, for the run JSON
        (CLAUDE.md rule 7) - enough to reproduce or audit a run's scaling
        without repeating the fit.
        """
        self._check_fitted()
        return {
            "columns": list(self._columns),
            "mean": [float(v) for v in self.mean_],
            "std": [float(v) for v in self.std_],
        }

    @classmethod
    def from_dict(cls, state):
        scaler = cls()
        scaler._columns = list(state["columns"])
        scaler.mean_ = np.array(state["mean"], dtype=float)
        scaler.std_ = np.array(state["std"], dtype=float)
        scaler._fitted = True
        return scaler
