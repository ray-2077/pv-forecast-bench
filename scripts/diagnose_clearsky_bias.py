"""Diagnose whether array11's k_ghi bias (median ~1.02, see
validate_clearsky.py output) is a uniform clear-sky model bias or a
solar-position timing error. The two need different fixes: a uniform bias
is a Linke-turbidity/model calibration issue, a timing error means the
midpoint-evaluation logic in src/data/clearsky.py is wrong. This script
only diagnoses which one it is - it does not calibrate or fix anything.

TRAINING YEARS ONLY (2012, 2013). 2014 (validation) and 2015 (test) are
never loaded here. Deciding how to model turbidity or timing by looking at
validation/test data first would be tuning on those splits before a single
split-respecting experiment has run - exactly what CLAUDE.md rule 3
(fit only on training data) and the "touch test once, at the end" rule
forbid. So this diagnosis is trained-eyes-only, same as any other fit.

Usage:
    python scripts/diagnose_clearsky_bias.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.clearsky import (
    add_clearsky,
    add_clearsky_index_ghi,
    add_daylight_mask,
    add_solar_position,
)

DATA_PATH = REPO_ROOT / "data" / "processed" / "array11_polySi_hourly.parquet"
TRAIN_YEARS = (2012, 2013)


def select_clear_hours(daylight):
    """Daylight hours belonging to "clear" days: daily (daylight-hour)
    mean k_ghi in the top decile of that calendar month's days.

    "That month" is the specific year-month a day falls in (Jan 2012 is
    compared only against other Jan 2012 days, not pooled with Jan 2013),
    so the decile threshold tracks each month's own turbidity/season
    rather than an average across two different Januaries.
    """
    daily_k_ghi = daylight.groupby(daylight.index.date)["k_ghi"].mean().dropna()
    daily_k_ghi.index = pd.to_datetime(daily_k_ghi.index)

    month = daily_k_ghi.index.to_period("M")
    threshold = daily_k_ghi.groupby(month).transform(lambda s: s.quantile(0.9))
    clear_daily_k_ghi = daily_k_ghi[daily_k_ghi >= threshold]

    clear_dates = clear_daily_k_ghi.index.date
    clear_hours = daylight[np.isin(daylight.index.date, clear_dates)]
    return clear_hours, clear_daily_k_ghi


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    # Training years only - see module docstring for why 2014/2015 are
    # never read into this script at all.
    df = df[df.index.year.isin(TRAIN_YEARS)]

    df = add_solar_position(df)
    df = add_clearsky(df)
    df = add_daylight_mask(df)
    df = add_clearsky_index_ghi(df)

    daylight = df[df["is_daylight"]]
    clear_hours, clear_daily_k_ghi = select_clear_hours(daylight)

    # 1. mean and count of k_ghi by hour of day, clear hours only
    print("1. k_ghi by hour of day, clear hours only:")
    by_hour = clear_hours.groupby(clear_hours.index.hour)["k_ghi"].agg(
        ["mean", "count"]
    )
    by_hour.index.name = "hour"
    print(by_hour.round(3).to_string())

    # 2. morning vs afternoon k_ghi, matched into 10-degree elevation bins
    print(
        "\n2. Morning (azimuth < 180) vs afternoon (azimuth >= 180) k_ghi,"
        "\n   matched into 10-degree solar elevation bins:"
    )
    elev_bins = np.arange(10, 100, 10)
    binned = clear_hours.copy()
    binned["elev_bin"] = pd.cut(binned["solar_elevation"], bins=elev_bins)
    binned["period"] = np.where(binned["solar_azimuth"] < 180, "morning", "afternoon")
    pivot = binned.groupby(["elev_bin", "period"], observed=True)["k_ghi"].mean()
    pivot = pivot.unstack("period")
    pivot["difference"] = pivot["afternoon"] - pivot["morning"]
    print(pivot.round(3).to_string())

    # 3. single clearest day: the clear day whose daylight-mean k_ghi is
    # closest to the median of the clear-day pool. Median rather than max
    # avoids picking a day where k_ghi is high because of cloud-edge
    # enhancement rather than genuinely clear sky.
    target = clear_daily_k_ghi.median()
    clearest_ts = (clear_daily_k_ghi - target).abs().idxmin()
    clearest_date = clearest_ts.date()
    print(
        f"\n3. Single clearest day: {clearest_date} "
        f"(daylight-mean k_ghi = {clear_daily_k_ghi.loc[clearest_ts]:.3f}, "
        f"clear-day pool median = {target:.3f})"
    )
    day_mask = (df.index.date == clearest_date) & df["is_daylight"]
    day_df = df.loc[
        day_mask,
        ["solar_elevation", "Global_Horizontal_Radiation", "ghi_cs", "k_ghi"],
    ].copy()
    day_df.insert(0, "hour", day_df.index.hour)
    day_df = day_df.set_index("hour")
    print(day_df.round(3).to_string())

    # 4. hour of max measured GHI vs hour of max ghi_cs, averaged over
    # clear days (search restricted to each day's daylight hours)
    print(
        "\n4. Hour of max measured GHI vs hour of max ghi_cs, "
        "averaged over clear days:"
    )
    clear_dates = np.unique(clear_hours.index.date)
    ghi_max_hours = []
    cs_max_hours = []
    for d in clear_dates:
        day = daylight[daylight.index.date == d]
        ghi_max_hours.append(day["Global_Horizontal_Radiation"].idxmax().hour)
        cs_max_hours.append(day["ghi_cs"].idxmax().hour)
    mean_ghi_hour = np.mean(ghi_max_hours)
    mean_cs_hour = np.mean(cs_max_hours)
    print(f"  clear days: {len(clear_dates)}")
    print(f"  mean hour of max measured GHI: {mean_ghi_hour:.2f}")
    print(f"  mean hour of max ghi_cs:       {mean_cs_hour:.2f}")
    print(f"  difference:                    {mean_ghi_hour - mean_cs_hour:.2f}")


if __name__ == "__main__":
    main()
