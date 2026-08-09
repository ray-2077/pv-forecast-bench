"""Diagnose the array17 val-to-test skill_vs_convex shift: candidate 3,
target-series difficulty.

CONTEXT (see scripts/diagnose_array17_test_shift.py for candidates 1 and
2, both inconclusive): array17's test skill_vs_convex is lower than val
in all 15 model x horizon cells; array11 and array12 both move the OTHER
way in all 18 of their cells. The outage window and clear-sky-index
(k_p) drift do not explain the direction. This script checks a third,
different kind of explanation: not a protocol or data artifact, but a
genuine change in how hard the target series itself is to forecast.

For each array, for 2014 (val) and 2015 (test) separately, daylight
hours only:
  - std(k_p)
  - mean |k_p(t) - k_p(t-1)|   (1-hour lag)
  - mean |k_p(t) - k_p(t-3)|   (3-hour lag, matches h=3)
  - mean |k_p(t) - k_p(t-6)|   (6-hour lag, matches h=6)
  - sky-class proportions (clear / partly_cloudy / overcast), via
    src.eval.sky.classify_sky - the SAME classifier and thresholds used
    for the paper's RQ3 stratification, not a new metric invented here.

Each lagged difference is computed on the full (day+night) hourly k_p
series so the time alignment across the shift is exact, then restricted
to rows where BOTH t and t-lag are daylight - this measures genuine
intra-day sky variability at exactly the horizon a model would have to
forecast across, not an overnight jump or a dawn/dusk artifact.

INTERPRETATION GUARD: all three arrays share ONE weather station
(CLAUDE.md "Data window" note). k_ghi-based sky class proportions are
therefore expected to be IDENTICAL across arrays for the same year -
printed per array only to confirm that (a sanity check on this script,
not a finding). k_p volatility, by contrast, is array-specific (it is
Active_Power / that array's own fitted clear-sky power), so a
DIVERGENCE in k_p volatility across arrays - array17 moving one way
while array11/array12 move the other - is NOT attributable to shared
weather and would itself need a further explanation if found.

Uses the same pipeline as scripts/diagnose_array17_test_shift.py: load
val (2014) and test (2015) with p_cs/k_p added, gain fit on TRAIN
(2011-2013) only and applied unchanged - test's clear-sky power via a
locally duplicated add_clearsky_power_to_test (see that script for why
this is not imported from scripts/run_final_test.py).

No fixes, no plotting, no changes to src/. Diagnosis only.

Usage:
    python scripts/diagnose_array17_target_volatility.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.clearsky_power import (
    add_clearsky_power,
    fit_gain,
    fit_temperature_climatology,
    model_clearsky_power,
)
from src.data.pipeline import add_clearsky_power_per_split, load_and_prepare
from src.data.splits import split_chronological
from src.eval.exclusions import exclusion_mask
from src.eval.sky import CATEGORIES, classify_sky, sky_class_counts

PROCESSED_DIR = REPO_ROOT / "data" / "processed"

ARRAYS = ["array11", "array12", "array17"]
LAGS = (1, 3, 6)


def add_clearsky_power_to_test(train, test, nameplate_kw, gamma_pdc):
    """Duplicated from scripts/diagnose_array17_test_shift.py /
    scripts/build_table6_dm.py - see either for the rationale (train-only
    fit applied unchanged to test; not imported from
    scripts/run_final_test.py, which is write-once).
    """
    temp_clim = fit_temperature_climatology(train)
    p_cs_raw_train = model_clearsky_power(train.index, nameplate_kw, gamma_pdc, temp_clim)
    gain, _n_gain_hours, _gain_iqr = fit_gain(train, p_cs_raw_train)

    p_cs_raw_test = model_clearsky_power(test.index, nameplate_kw, gamma_pdc, temp_clim)
    return add_clearsky_power(test, p_cs_raw_test, gain, nameplate_kw)


def load_val_test(array):
    """val (2014) and test (2015), both with p_cs/k_p added - val via
    add_clearsky_power_per_split, test via add_clearsky_power_to_test
    above, both using the SAME train(2011-2013)-only fitted gain/temp_clim.
    """
    df, nameplate_kw, gamma_pdc = load_and_prepare(array, PROCESSED_DIR)
    train, val, test = split_chronological(df)
    train, val, _gain_info = add_clearsky_power_per_split(train, val, test, nameplate_kw, gamma_pdc)
    test = add_clearsky_power_to_test(train, test, nameplate_kw, gamma_pdc)
    return val, test


def volatility_stats(df_year, array, exclude_outage):
    """std(k_p) and mean |k_p(t) - k_p(t-lag)| for lag in LAGS, daylight
    hours only (both t and t-lag must be daylight for a lagged diff to
    count - see module docstring).

    exclude_outage=True additionally drops any hour inside
    src.eval.exclusions.KNOWN_OUTAGES for this array from BOTH the
    std(k_p) population and both endpoints of every lagged pair -
    array17's 2015-06-05..09 outage puts k_p near zero for equipment
    reasons, not sky reasons, and would otherwise inflate array17's 2015
    volatility numbers with a data-quality artifact rather than genuine
    target difficulty. exclude_outage=False reproduces the raw,
    uncorrected numbers for comparison - the gap between the two IS
    itself diagnostic (see main()).
    """
    daylight_mask = df_year["is_daylight"]
    if exclude_outage:
        daylight_mask = daylight_mask & ~exclusion_mask(array, df_year.index)
    k_p = df_year["k_p"]

    stats = {"std_k_p": k_p[daylight_mask].std(), "n_daylight_hours": int(daylight_mask.sum())}

    for lag in LAGS:
        shifted_k_p = k_p.shift(lag)
        shifted_daylight = daylight_mask.shift(lag).fillna(False).astype(bool)
        both_daylight = daylight_mask & shifted_daylight
        diff = (k_p - shifted_k_p).abs()
        stats[f"mean_abs_dk_p_lag{lag}h"] = diff[both_daylight].mean()
        stats[f"n_pairs_lag{lag}h"] = int(both_daylight.sum())

    return stats


def sky_proportions(df_year):
    sky = classify_sky(df_year)
    counts = sky_class_counts(sky)
    total = int(counts.sum())
    props = (counts / total) if total else counts.astype(float)
    return counts, props, total


def _build_vol_df(exclude_outage):
    vol_rows = []
    for array in ARRAYS:
        val, test = load_val_test(array)
        for year_label, df_year in [("2014 (val)", val), ("2015 (test)", test)]:
            v = volatility_stats(df_year, array, exclude_outage=exclude_outage)
            v["array"] = array
            v["year"] = year_label
            vol_rows.append(v)
    return pd.DataFrame(vol_rows).set_index(["array", "year"])


def _delta_df(vol_df):
    delta_rows = []
    for array in ARRAYS:
        val_row = vol_df.loc[(array, "2014 (val)")]
        test_row = vol_df.loc[(array, "2015 (test)")]
        delta_rows.append(
            {
                "array": array,
                "d_std_k_p": test_row["std_k_p"] - val_row["std_k_p"],
                "d_mean_abs_dk_p_lag1h": test_row["mean_abs_dk_p_lag1h"] - val_row["mean_abs_dk_p_lag1h"],
                "d_mean_abs_dk_p_lag3h": test_row["mean_abs_dk_p_lag3h"] - val_row["mean_abs_dk_p_lag3h"],
                "d_mean_abs_dk_p_lag6h": test_row["mean_abs_dk_p_lag6h"] - val_row["mean_abs_dk_p_lag6h"],
            }
        )
    return pd.DataFrame(delta_rows).set_index("array")


def main():
    print("=" * 100)
    print("CANDIDATE 3: target-series difficulty - k_p volatility and sky-class mix, val (2014) vs test (2015)")
    print("=" * 100)

    vol_cols = ["std_k_p", "mean_abs_dk_p_lag1h", "mean_abs_dk_p_lag3h", "mean_abs_dk_p_lag6h", "n_daylight_hours"]

    print(
        "\n--- k_p volatility, daylight hours only, OUTAGE-EXCLUDED "
        "(array17's 2015-06-05..09 dropped from both endpoints of every lagged "
        "pair - equipment-off k_p is not sky difficulty; see module docstring) ---\n"
    )
    vol_df_clean = _build_vol_df(exclude_outage=True)
    print(vol_df_clean[vol_cols].round(4).to_string())

    print("\n--- val -> test change per array, OUTAGE-EXCLUDED (test - val; positive = MORE volatile on test) ---\n")
    print(_delta_df(vol_df_clean).round(4).to_string())

    print(
        "\n--- same table, RAW (outage NOT excluded) - for comparison only, shows "
        "how much of any array17 change is the outage itself rather than sky ---\n"
    )
    vol_df_raw = _build_vol_df(exclude_outage=False)
    print(vol_df_raw[vol_cols].round(4).to_string())
    print("\n--- val -> test change per array, RAW ---\n")
    print(_delta_df(vol_df_raw).round(4).to_string())

    sky_rows = []
    for array in ARRAYS:
        val, test = load_val_test(array)
        for year_label, df_year in [("2014 (val)", val), ("2015 (test)", test)]:
            counts, props, total = sky_proportions(df_year)
            row = {"array": array, "year": year_label, "n_daylight_classified": total}
            for cat in CATEGORIES:
                row[f"{cat}_n"] = int(counts[cat])
                row[f"{cat}_frac"] = round(float(props[cat]), 4)
            sky_rows.append(row)

    sky_df = pd.DataFrame(sky_rows).set_index(["array", "year"])
    frac_cols = [f"{cat}_frac" for cat in CATEGORIES]
    n_cols = [f"{cat}_n" for cat in CATEGORIES]

    print("\n--- sky-class proportions (k_ghi-based, src.eval.sky.classify_sky - shared weather station) ---\n")
    print(sky_df[["n_daylight_classified"] + n_cols + frac_cols].round(4).to_string())

    print(
        "\nSanity check (see module docstring): rows for the SAME year should be "
        "identical (or near-identical) across all three arrays, since sky class "
        "depends only on the shared weather station's k_ghi, not on any array's own "
        "power reading. A mismatch here would indicate a bug in this script, not a "
        "real per-array sky difference."
    )


if __name__ == "__main__":
    main()
