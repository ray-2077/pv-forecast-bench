"""Validate src/data/clearsky.py against array11 (poly-Si).

Loads the processed hourly parquet, adds solar position, clear-sky
irradiance, the daylight mask, and the clear-sky GHI index, then prints
diagnostics only - no plotting, no files written.

Usage:
    python scripts/validate_clearsky.py
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

DATA_PATH = REPO_ROOT / "data" / "processed" / "array11_polySi_hourly.parquet"


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    df = add_solar_position(df)
    df = add_clearsky(df)
    df = add_daylight_mask(df)
    df = add_clearsky_index_ghi(df)

    daylight = df[df["is_daylight"]]

    # 1. count and percentage of daylight hours
    n_daylight = len(daylight)
    pct_daylight = 100.0 * n_daylight / len(df)
    print(f"Daylight hours: {n_daylight} of {len(df)} ({pct_daylight:.2f}%)")

    # 2. k_ghi percentiles, daylight hours
    print("\nk_ghi percentiles (daylight hours):")
    qs = daylight["k_ghi"].quantile([0.5, 0.9, 0.95, 0.99])
    for q, val in qs.items():
        print(f"  p{int(round(q * 100))}: {val:.3f}")

    # 3. mean measured GHI and mean ghi_cs by hour of day, daylight hours
    print("\nMean GHI vs ghi_cs by hour of day (daylight hours):")
    by_hour = daylight.groupby(daylight.index.hour)[
        ["Global_Horizontal_Radiation", "ghi_cs"]
    ].mean()
    by_hour.index.name = "hour"
    print(by_hour.round(1).to_string())

    # 4. 10 clearest days in 2015: daylight-hour mean k_ghi closest to the
    # median of the top decile of daily daylight-hour mean k_ghi.
    daylight_2015 = daylight[daylight.index.year == 2015]
    daily_k_ghi = (
        daylight_2015.groupby(daylight_2015.index.date)["k_ghi"].mean().dropna()
    )

    top_decile_threshold = daily_k_ghi.quantile(0.9)
    top_decile = daily_k_ghi[daily_k_ghi >= top_decile_threshold]
    target = top_decile.median()

    closest10_idx = (daily_k_ghi - target).abs().sort_values().head(10).index
    clearest_days = daily_k_ghi.loc[closest10_idx].sort_index()

    print(f"\nTop-decile threshold k_ghi: {top_decile_threshold:.3f}")
    print(f"Target (median of top decile): {target:.3f}")
    print("\n10 clearest days in 2015 (daylight-hour mean k_ghi closest to target):")
    print(clearest_days.round(3).to_string())

    example_day = clearest_days.index[0]
    print(f"\nHourly measured GHI vs ghi_cs on {example_day} (daylight hours):")
    day_mask = (df.index.date == example_day) & df["is_daylight"]
    day_df = df.loc[day_mask, ["Global_Horizontal_Radiation", "ghi_cs"]]
    print(day_df.round(1).to_string())


if __name__ == "__main__":
    main()
