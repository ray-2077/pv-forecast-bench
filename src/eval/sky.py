"""Sky-condition classification, for stratifying forecast errors by
atmospheric condition (RQ3) - clear / partly cloudy / overcast.

Classified from k_ghi (clear-sky index of GHI) and its short-window
variability ONLY. Never from k_p (clear-sky index of POWER), for two
reasons that must both hold for this classification to be a legitimate
stratification variable:

(a) Sky condition is a property of the ATMOSPHERE, and DKASC's three
    evaluated arrays (11, 12, 17 - see CLAUDE.md "Data window") are
    co-located and share one weather station. k_ghi is common to all
    three. k_p is array-specific (different inverter efficiency, gain,
    degradation per array - see src.data.clearsky_power), so classifying
    by k_p would give three different "sky conditions" for the same hour
    of the same sky, which makes no physical sense and would silently
    let array-level effects leak into what is supposed to be a
    sky-condition comparison.

(b) k_p is computed from measured Active_Power - the forecasting TARGET.
    Stratifying forecast errors by a classification derived from the
    target conditions each error on the very quantity being predicted,
    which biases the comparison in an unpredictable direction. k_ghi
    conditions on the atmosphere instead, which is legitimate: the
    forecast does not get to see k_ghi at the target time either (a model
    is never fed oracle_ weather at target time in the 'lagged' regime,
    src.features.build), but an ANALYST is allowed to look at it after
    the fact to ask "how did this model do on clear vs. overcast hours".

IMPORTANT - target-time information, but only for analysis: classify_sky
uses k_ghi AT THE TARGET TIME t (via a window that ends at t), not at any
issue time t-h. That is only legitimate because this is a post-hoc
STRATIFICATION of already-computed forecast errors, never a model input.
It must never be passed into src.features.build.build_features, and no
column produced here appears in that module's feature_names() - if it
ever needs to, that would be an oracle-regime feature and must be named
and gated exactly as src.features.build already gates oracle_ weather.

THRESHOLDS: 0.75 / 0.10 / 0.40 below are conventional starting values
carried over from the general sky-classification literature, not fitted
to this site or tuned on any split - doing so would violate CLAUDE.md's
"do not tune on the test set" rule in spirit even if applied only to
validation, since a threshold search is itself a fit. Report the
resulting class proportions (see sky_class_counts) to judge whether they
are sensible for DKASC Alice Springs. In particular, this site's k_ghi is
NOT calibrated to 1.0 on clear days - median k_ghi is approx 1.02 (see
paper/PROJECT_CHECKPOINT.md Finding 1) - so these thresholds should be
read as relative cut points against that uncalibrated scale, not as
claims about a physically perfect clear-sky reference.
"""

import numpy as np
import pandas as pd

from src.features.build import _last_n_valid_obs_stat

CLEAR_MEAN_MIN = 0.75
CLEAR_STD_MAX = 0.10
OVERCAST_MEAN_MAX = 0.40

CATEGORIES = ["clear", "partly_cloudy", "overcast"]

DEFAULT_WINDOW = 3


def classify_sky(df, window=DEFAULT_WINDOW):
    """Classify each row of df into a sky condition, from k_ghi alone.

    For every row, computes k_ghi_mean and k_ghi_std over the last
    `window` VALID (non-NaN) k_ghi observations, inclusive of the current
    row - reusing src.features.build's last-N-valid-observations pattern
    (_last_n_valid_obs_stat) rather than a wall-clock window, for the
    same reason documented there: k_ghi is undefined at night, so a
    wall-clock window ending near sunrise/sunset can be empty even though
    daylight observations exist further back. Unlike build.py's use of
    that helper, NO horizon shift is applied here: this is a
    target-time analysis variable, not a model feature (see module
    docstring).

    Classification, in order:
      k_ghi_mean >= CLEAR_MEAN_MIN and k_ghi_std < CLEAR_STD_MAX -> 'clear'
      k_ghi_mean < OVERCAST_MEAN_MAX                             -> 'overcast'
      otherwise                                                  -> 'partly_cloudy'
    (a row with too little history for k_ghi_std to be defined yet just
    fails the 'clear' test - NaN < CLEAR_STD_MAX is False - and falls
    through to 'overcast' or 'partly_cloudy' by k_ghi_mean alone, same as
    any other row that is not clear)

    Returns a pandas Categorical Series (categories = CATEGORIES), aligned
    to df.index, NaN at night (is_daylight False) or on the rare row with
    no k_ghi history at all yet (k_ghi_mean itself undefined - only
    possible at the very start of a series).

    Requires df to carry k_ghi (src.data.clearsky.add_clearsky_index_ghi)
    and is_daylight (src.data.clearsky.add_daylight_mask).
    """
    if "k_ghi" not in df.columns:
        raise KeyError(
            "classify_sky requires column 'k_ghi'; run "
            "src.data.clearsky.add_clearsky_index_ghi on df first"
        )
    if "is_daylight" not in df.columns:
        raise KeyError(
            "classify_sky requires column 'is_daylight'; run "
            "src.data.clearsky.add_daylight_mask on df first"
        )

    k_ghi_mean = _last_n_valid_obs_stat(df["k_ghi"], window, "mean")
    k_ghi_std = _last_n_valid_obs_stat(df["k_ghi"], window, "std")

    mean_vals = k_ghi_mean.to_numpy()
    std_vals = k_ghi_std.to_numpy()

    is_clear = (mean_vals >= CLEAR_MEAN_MIN) & (std_vals < CLEAR_STD_MAX)
    is_overcast = mean_vals < OVERCAST_MEAN_MAX

    labels = np.where(is_clear, "clear", np.where(is_overcast, "overcast", "partly_cloudy"))
    sky = pd.Series(labels, index=df.index, dtype=object)
    sky = sky.mask(np.isnan(mean_vals))
    sky = sky.where(df["is_daylight"].to_numpy())
    sky = sky.astype(pd.CategoricalDtype(categories=CATEGORIES))
    sky.name = "sky_class"
    return sky


def sky_class_counts(sky):
    """Count of daylight rows in each CATEGORIES class (NaN/night rows
    excluded), as a pandas Series indexed by CATEGORIES in that fixed
    order, zero-filled for a class that does not occur.
    """
    counts = sky.value_counts(dropna=True)
    return counts.reindex(CATEGORIES, fill_value=0).astype(int)
