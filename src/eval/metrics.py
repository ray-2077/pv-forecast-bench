"""Forecast accuracy metrics.

All functions accept 1D numpy arrays or pandas Series for y_true / y_pred
(and y_ref where relevant) and return plain Python floats. No plotting, no
model code, no file writing - this module is pure and deterministic.

NaN handling: every function drops NaN pairs (or triples, for the
functions that also take y_ref) using the SAME mask across all of its
inputs before computing anything, so that e.g. mae() and rmse() called on
the same inputs are computed over identical samples, and so that
skill_score()'s numerator and denominator RMSEs are computed over the
same subset of timestamps rather than each silently dropping a different
set of NaNs.
"""

import numpy as np


def _align_drop_nan(*arrays):
    """Convert inputs to float arrays and drop any index where ANY of the
    arrays is NaN. Returns one cleaned array per input, in the same order.
    """
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    mask = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        mask &= ~np.isnan(a)
    return tuple(a[mask] for a in arrays)


def mae(y_true, y_pred):
    y_true, y_pred = _align_drop_nan(y_true, y_pred)
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true, y_pred):
    y_true, y_pred = _align_drop_nan(y_true, y_pred)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mbe(y_true, y_pred):
    """Mean bias error, y_pred minus y_true. Positive means over-forecasting."""
    y_true, y_pred = _align_drop_nan(y_true, y_pred)
    return float(np.mean(y_pred - y_true))


def nrmse(y_true, y_pred, nameplate_kw):
    """RMSE divided by nameplate capacity, as a percentage.

    Normalising by nameplate capacity rather than by mean observed power
    so that nRMSE is comparable across arrays of different sizes (this
    project has 5.0, 5.1, and 7.0 kW arrays) - normalising by mean power
    would instead reward/penalise arrays for their average output level,
    which has nothing to do with forecast quality.
    """
    return 100.0 * rmse(y_true, y_pred) / nameplate_kw


def skill_score(y_true, y_pred, y_ref):
    """1 - rmse(y_true, y_pred) / rmse(y_true, y_ref).

    Positive means better than the reference forecast, negative means
    worse. Raises ValueError if the reference RMSE is zero (division
    undefined - a perfect reference forecast makes skill meaningless).
    """
    y_true, y_pred, y_ref = _align_drop_nan(y_true, y_pred, y_ref)
    ref_rmse = rmse(y_true, y_ref)
    if ref_rmse == 0:
        raise ValueError("skill_score: reference RMSE is zero, cannot divide")
    return 1.0 - rmse(y_true, y_pred) / ref_rmse


def all_metrics(y_true, y_pred, y_ref, nameplate_kw):
    """Return {mae, rmse, nrmse, mbe, skill, n_samples}, all computed on
    the identical set of timestamps (rows where none of y_true, y_pred,
    y_ref is NaN).
    """
    y_true, y_pred, y_ref = _align_drop_nan(y_true, y_pred, y_ref)
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred, nameplate_kw),
        "mbe": mbe(y_true, y_pred),
        "skill": skill_score(y_true, y_pred, y_ref),
        "n_samples": int(len(y_true)),
    }
