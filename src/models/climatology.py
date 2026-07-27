"""Two reference forecasters that do NOT rely on a recent observation of
Active_Power/k_p, unlike SmartPersistence.

Motivation: scripts/diagnose_baseline_error.py showed SmartPersistence has
MBE up to -2.15 kW at midday for h=6 on array11 (2014) - a midday target
at that horizon is issued near dawn, so the k_p it persists forward is
stale (often forward-filled from the previous evening). A skill score
measured against a baseline that is structurally broken at particular
hours is inflated: XGBoost only has to beat a bad number there, not
forecast well. Yang et al. (2020, Solar Energy 210:20-37) recommend the
RMSE skill score based on the optimal convex combination of climatology
and persistence as the standard reference forecast, precisely because
persistence alone can be this misleading. ConvexCombination below
implements that recommendation; Climatology is its other ingredient and
is also useful as a reference in its own right.

Both classes implement BaseForecaster (src/models/base.py) with the same
fit/predict interface as SmartPersistence (src/models/persistence.py, not
modified here), so the runner and eval code need no changes to use them.
"""

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster
from src.models.persistence import SmartPersistence

MIN_OBS_PER_CELL = 10

# Grid of candidate weights for ConvexCombination.fit, step 0.01. A closed
# form exists (least-squares over one scalar), but the grid is easier to
# read and verify, and 101 evaluations costs nothing next to fitting two
# models.
_W_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 2)


class Climatology(BaseForecaster):
    """Predicts the TRAINING mean k_p for a target time's (month,
    hour-of-day) cell, times that target time's p_cs. Uses no
    observation of the array at all - not the most recent one, not any
    from earlier the same day - only the calendar position of the target
    time and the deterministic p_cs there. Requires df to carry k_p (fit)
    and p_cs (predict), i.e. src.data.clearsky_power.add_clearsky_power
    has already run, with gain fit on TRAINING data only.
    """

    name = "climatology"

    def __init__(self):
        # pd.Series, MultiIndex (month, hour) -> mean k_p over training
        # years. Set by fit().
        self._table = None

    def fit(self, df_train: pd.DataFrame, horizon: int) -> "Climatology":
        if "k_p" not in df_train.columns:
            raise KeyError(
                "Climatology.fit requires column 'k_p'; run "
                "src.data.clearsky_power.add_clearsky_power on df_train first"
            )

        month = pd.Index(df_train.index.month, name="month")
        hour = pd.Index(df_train.index.hour, name="hour")
        grouped = df_train["k_p"].groupby([month, hour])

        counts = grouped.count()  # non-NaN count per (month, hour) cell
        means = grouped.mean()

        # Cells with zero observations are hours that are never daylight
        # in that month (k_p is NaN by construction at night - see
        # add_clearsky_power) and are expected, not an error: they simply
        # never appear as a lookup target for a daylight prediction.
        # Cells with SOME but too few observations are the actual
        # problem - too little training signal to trust a mean from -
        # and are what this check catches.
        insufficient = counts[(counts > 0) & (counts < MIN_OBS_PER_CELL)]
        if not insufficient.empty:
            raise ValueError(
                f"Climatology.fit: {len(insufficient)} (month, hour) cell(s) "
                f"have fewer than {MIN_OBS_PER_CELL} valid k_p observations "
                f"in training data: {list(insufficient.index)[:5]}"
                f"{' ...' if len(insufficient) > 5 else ''}"
            )

        self._table = means
        return self

    def predict(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        # horizon is accepted only to satisfy the BaseForecaster
        # interface and is otherwise unused: the forecast is
        # mean_k_p[month(t), hour(t)] * p_cs(t), neither of which depends
        # on how far in advance the forecast is issued. Climatology is
        # the same 1h-ahead as it is 24h-ahead.
        if self._table is None:
            raise RuntimeError("call fit() before predict()")
        if "p_cs" not in df.columns:
            raise KeyError(
                "Climatology.predict requires column 'p_cs'; run "
                "src.data.clearsky_power.add_clearsky_power on df first"
            )

        keys = pd.MultiIndex.from_arrays(
            [df.index.month, df.index.hour], names=["month", "hour"]
        )
        mean_k_p = self._table.reindex(keys).to_numpy()

        preds = pd.Series(mean_k_p, index=df.index) * df["p_cs"].to_numpy()
        preds = preds.clip(lower=0).dropna()
        preds.name = "p_hat"
        return preds


class ConvexCombination(BaseForecaster):
    """w * SmartPersistence + (1 - w) * Climatology, with w in [0, 1]
    chosen to minimise RMSE on the VALIDATION split - the reference
    forecast recommended by Yang et al. (2020, Solar Energy 210:20-37).
    See the module docstring for why: persistence alone can be
    structurally worse than climatology at specific hours (e.g. the
    dawn-issue-time/midday-target case), and a convex combination lets the
    fit discover that per horizon rather than assuming persistence is
    always the harder baseline to beat.
    """

    name = "convex_reference"

    def __init__(self):
        self.w = None
        self._climatology = None
        self._persistence = None

    def fit(self, df_train: pd.DataFrame, horizon: int, df_val: pd.DataFrame) -> "ConvexCombination":
        self._climatology = Climatology().fit(df_train, horizon)
        self._persistence = SmartPersistence().fit(df_train, horizon)  # no-op, see persistence.py

        # CRITICAL: w is fit by grid search over df_val ONLY. df_val is
        # the sole data this loop ever reads - there is no path here that
        # could touch a test split, no matter what the caller later passes
        # to predict(). Per CLAUDE.md rule 6/research-integrity: fitting
        # this weight on test would let the reference forecast itself
        # leak test-set information into every skill score computed
        # against it.
        clim_val = self._climatology.predict(df_val, horizon)
        pers_val = self._persistence.predict(df_val, horizon)
        common_idx = clim_val.index.intersection(pers_val.index)

        y_val = df_val.loc[common_idx, "Active_Power"].to_numpy()
        clim_vals = clim_val.loc[common_idx].to_numpy()
        pers_vals = pers_val.loc[common_idx].to_numpy()

        best_w, best_rmse = 0.0, float("inf")
        for w in _W_GRID:
            blend = w * pers_vals + (1.0 - w) * clim_vals
            candidate_rmse = float(np.sqrt(np.mean((blend - y_val) ** 2)))
            if candidate_rmse < best_rmse:
                best_rmse = candidate_rmse
                best_w = float(w)

        self.w = best_w
        return self

    def predict(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        if self.w is None:
            raise RuntimeError("call fit() before predict()")

        clim_preds = self._climatology.predict(df, horizon)
        pers_preds = self._persistence.predict(df, horizon)
        # Only blend where BOTH components produced a prediction - see
        # base class convention. Climatology can predict at any target
        # time; persistence cannot until horizon hours of history exist,
        # so this intersection is what actually restricts the combined
        # forecast's coverage.
        common_idx = clim_preds.index.intersection(pers_preds.index)

        blend = self.w * pers_preds.loc[common_idx] + (1.0 - self.w) * clim_preds.loc[common_idx]
        # Both components are already clipped at 0 individually, so a
        # convex combination of them is already >= 0 - this clip is a
        # no-op safety net, not load-bearing.
        blend = blend.clip(lower=0)
        blend.name = "p_hat"
        return blend
