"""NOTE: run against the ORIGINAL 2009-2013 training window, before
TRAIN_YEARS was narrowed to 2011-2013 and array07 was dropped entirely (see
CLAUDE.md "Data window"). Retained unchanged for provenance - this script IS
the diagnosis that led to array07's exclusion, so its own local TRAIN_YEARS
below deliberately still reflects the window it was actually run under. Do
not re-run and compare against current results.

Diagnose whether array07's negative XGBoost skill and near-1.0 convex
weight are caused by a level shift in array07's output between the training
years and 2014, rather than by weather.

EVIDENCE MOTIVATING THIS (results/reference_comparison.csv, 2014 val):
convex_weight is 0.99/0.96/0.90 for array07 at h=1/3/6, but 0.77/0.25/0.01
for array11 and 0.77/0.28/0.04 for array12. Climatology and XGBoost both
learn a level from 2009-2013 and apply it unchanged to 2014; persistence
uses a recent observation and self-corrects every hour instead. A convex
weight near 1.0 means climatology is contributing almost nothing useful for
array07 - consistent with array07's output level having shifted under the
climatology's feet while it stayed put under persistence's.

This script checks that directly with a performance ratio - mean(Active_Power)
/ mean(GHI), daylight hours only - which normalises out inter-annual weather
variation (a cloudier or clearer year moves both the numerator and the
denominator together). A trend or step in this ratio is the array itself
changing; a trend or step in raw mean Active_Power that vanishes once
divided by GHI is just the weather.

Do NOT load 2015 (test). This script loads and inspects 2009-2014 only.
Deciding how to interpret this diagnosis by looking at 2015 data before a
single split-respecting experiment has run on it would be tuning on the
test set - exactly what CLAUDE.md's "touch test once, at the end" rule
forbids. Everything needed to answer the question (has array07 shifted
level going into the validation year) lives in train (2009-2013) and val
(2014) alone.

Clear-sky power (p_cs, k_p) is fit and applied exactly as run_xgb_dev.py
does it: temperature climatology and gain fit on TRAIN years only
(2009-2013), then applied - unchanged - to both train and val. So k_p's
drift (if any) shows the array moving relative to a level fixed once on
2009-2013 data, the same way Climatology and ConvexCombination fix their
level once and carry it into 2014 unmodified.

No fixes, no plotting. Diagnosis only.

Usage:
    python scripts/diagnose_array_level_shift.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.clearsky import (
    add_clearsky,
    add_clearsky_index_ghi,
    add_daylight_mask,
    add_solar_position,
)
from src.data.clearsky_power import (
    GAMMA_PDC_CDTE,
    GAMMA_PDC_SILICON,
    add_clearsky_power,
    fit_gain,
    fit_temperature_climatology,
    model_clearsky_power,
)
from src.data.splits import split_chronological

PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Same three arrays, filenames, and nameplate/gamma_pdc values as
# scripts/run_xgb_dev.py and scripts/validate_persistence.py.
ARRAYS = {
    "array07": ("array07_CdTe_hourly.parquet", 7.0, GAMMA_PDC_CDTE),
    "array11": ("array11_polySi_hourly.parquet", 5.0, GAMMA_PDC_SILICON),
    "array12": ("array12_monoSi_hourly.parquet", 5.1, GAMMA_PDC_SILICON),
}

DIAGNOSIS_YEARS = (2009, 2010, 2011, 2012, 2013, 2014)
TRAIN_YEARS = (2009, 2010, 2011, 2012, 2013)
VAL_YEAR = 2014


def load_2009_2014(array_key):
    """Load one array's processed parquet, restricted to 2009-2014 before
    anything else touches it, then add solar position / clear-sky / daylight
    mask / k_ghi - the same steps run_xgb_dev.py's load_and_prepare uses,
    minus clear-sky power, which needs a train-only fit (below).
    """
    filename, nameplate_kw, gamma_pdc = ARRAYS[array_key]
    df = pd.read_parquet(PROCESSED_DIR / filename)
    df = df[df.index.year.isin(DIAGNOSIS_YEARS)]  # 2015 never enters this script

    df = add_solar_position(df)
    df = add_clearsky(df)
    df = add_daylight_mask(df)
    df = add_clearsky_index_ghi(df)
    return df, nameplate_kw, gamma_pdc


def add_p_cs_k_p(df, nameplate_kw, gamma_pdc):
    """Split into train (2009-2013) / val (2014), fit temperature
    climatology and gain on train only, apply that fit unchanged to both
    train and val, then recombine into a single 2009-2014 frame carrying
    p_cs and k_p. test_years=(2015,) is passed only to satisfy
    split_chronological's chronology check; no 2015 rows exist in df so
    the resulting test split is always empty.
    """
    train, val, _test = split_chronological(
        df, train_years=TRAIN_YEARS, val_years=(VAL_YEAR,), test_years=(2015,)
    )

    temp_clim = fit_temperature_climatology(train)

    p_cs_raw_train = model_clearsky_power(train.index, nameplate_kw, gamma_pdc, temp_clim)
    gain, n_gain_hours, gain_iqr = fit_gain(train, p_cs_raw_train)
    train = add_clearsky_power(train, p_cs_raw_train, gain, nameplate_kw)

    p_cs_raw_val = model_clearsky_power(val.index, nameplate_kw, gamma_pdc, temp_clim)
    val = add_clearsky_power(val, p_cs_raw_val, gain, nameplate_kw)

    combined = pd.concat([train, val]).sort_index()
    gain_info = {"gain": gain, "gain_n_hours": n_gain_hours, "gain_iqr": gain_iqr}
    return combined, gain_info


def yearly_stats(df):
    """Per-year, daylight-only: mean Active_Power, mean GHI, the
    weather-normalised performance ratio mean(Active_Power)/mean(GHI), mean
    k_p, and the daylight-hour count.
    """
    daylight = df[df["is_daylight"]]
    grouped = daylight.groupby(daylight.index.year)

    stats = pd.DataFrame(
        {
            "mean_active_power_kw": grouped["Active_Power"].mean(),
            "mean_ghi_wm2": grouped["Global_Horizontal_Radiation"].mean(),
            "mean_k_p": grouped["k_p"].mean(),
            "n_daylight_hours": grouped.size(),
        }
    )
    stats["performance_ratio"] = stats["mean_active_power_kw"] / stats["mean_ghi_wm2"]
    stats.index.name = "year"

    # Reorder to match the requested print order: AP, GHI, ratio, k_p, count.
    return stats[
        ["mean_active_power_kw", "mean_ghi_wm2", "performance_ratio", "mean_k_p", "n_daylight_hours"]
    ]


def monthly_performance_ratio(df, year):
    """Monthly mean performance ratio for one year, daylight hours only."""
    daylight = df[df["is_daylight"] & (df.index.year == year)]
    grouped = daylight.groupby(daylight.index.month)
    ratio = grouped["Active_Power"].mean() / grouped["Global_Horizontal_Radiation"].mean()
    ratio.index.name = "month"
    return ratio


def main():
    combined_by_array = {}

    for array_key in sorted(ARRAYS):
        df, nameplate_kw, gamma_pdc = load_2009_2014(array_key)
        combined, gain_info = add_p_cs_k_p(df, nameplate_kw, gamma_pdc)
        combined_by_array[array_key] = combined

        print(f"\n=== {array_key} (gain fit on train 2009-2013: {gain_info['gain']:.4f}, "
              f"n_hours={gain_info['gain_n_hours']}, IQR={gain_info['gain_iqr']:.4f}) ===")

        stats = yearly_stats(combined)
        print(stats.round(4).to_string())

        ratio_2009_2013_mean = stats.loc[2009:2013, "performance_ratio"].mean()
        ratio_2014 = stats.loc[VAL_YEAR, "performance_ratio"]
        pct_change = (ratio_2014 - ratio_2009_2013_mean) / ratio_2009_2013_mean * 100
        print(
            f"performance_ratio: 2009-2013 mean={ratio_2009_2013_mean:.4f}  "
            f"2014={ratio_2014:.4f}  pct_change={pct_change:+.2f}%"
        )

    print("\n=== array07 monthly performance ratio, 2013 vs 2014 ===")
    array07_df = combined_by_array["array07"]
    monthly_2013 = monthly_performance_ratio(array07_df, 2013)
    monthly_2014 = monthly_performance_ratio(array07_df, 2014)
    monthly = pd.DataFrame({"2013": monthly_2013, "2014": monthly_2014})
    monthly.index.name = "month"
    print(monthly.round(4).to_string())


if __name__ == "__main__":
    main()
