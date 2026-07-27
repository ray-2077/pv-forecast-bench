"""Smart persistence: the baseline every other model in this benchmark is
measured against (CLAUDE.md rule 4 - skill score vs this model is the
headline metric, not raw RMSE).

Forecast: P_hat(t) = k_p(t - horizon) * p_cs(t), where k_p is the clear-sky
power index (Active_Power / p_cs) and p_cs is the clear-sky power reference
from src.data.clearsky_power. In words: "the array will produce the same
fraction of its clear-sky potential at the target time as it did the last
time we observed it, horizon hours ago."
"""

import pandas as pd

from src.models.base import BaseForecaster

REQUIRED_COLS = ("k_p", "p_cs")


class SmartPersistence(BaseForecaster):
    """Persist the most recently observed clear-sky power index forward.

    Requires df to already carry k_p and p_cs (i.e.
    src.data.clearsky_power.add_clearsky_power has run on it upstream,
    with the clear-sky model and gain fitted on TRAINING data only). See
    the module docstring for the alignment convention: predict(df,
    horizon) returns predictions indexed by target time t, each using only
    k_p observed at t - horizon.
    """

    name = "smart_persistence"

    FFILL_LIMIT_HOURS = 24

    def __init__(self):
        # Bool Series, indexed like the most recent predict() call's
        # output, True where that prediction relied on a forward-filled
        # k_p rather than one observed exactly horizon hours earlier. Set
        # by predict(); read through the fallback_fraction property.
        self._fallback_mask = None

    def fit(self, df_train: pd.DataFrame, horizon: int) -> "SmartPersistence":
        # No-op: smart persistence has no free parameters to learn. The
        # clear-sky power model and its gain were already fitted on
        # training data upstream, in src.data.clearsky_power (fit_gain,
        # fit_temperature_climatology). There is nothing left to fit here.
        return self

    def predict(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise KeyError(
                f"SmartPersistence.predict requires columns {missing}; run "
                "src.data.clearsky_power.add_clearsky_power on df first"
            )

        # Positional shift on a regular hourly index: shifted.loc[t] ==
        # k_p.loc[t - horizon hours]. The first `horizon` rows become NaN
        # (no history yet), matching check_no_lookahead's requirement.
        k_p_shifted = df["k_p"].shift(horizon)

        # k_p is NaN at night (p_cs below 2% of nameplate - see
        # clearsky_power.add_clearsky_power). Forward-fill so a forecast
        # issued at night uses the most recent valid daytime k_p, capped
        # at 24h so a long data gap does not fill forever.
        k_p_filled = k_p_shifted.ffill(limit=self.FFILL_LIMIT_HOURS)

        preds = (k_p_filled * df["p_cs"]).clip(lower=0)
        preds = preds.dropna()
        preds.name = "p_hat"

        fallback_mask = k_p_shifted.isna() & k_p_filled.notna()
        self._fallback_mask = fallback_mask.reindex(preds.index)

        return preds

    @property
    def fallback_fraction(self) -> float:
        """Fraction of the most recent predict() call's predictions that
        used a forward-filled k_p rather than one observed exactly
        horizon hours before the target time.
        """
        if self._fallback_mask is None:
            raise RuntimeError("call predict() before reading fallback_fraction")
        if len(self._fallback_mask) == 0:
            return float("nan")
        return float(self._fallback_mask.mean())
