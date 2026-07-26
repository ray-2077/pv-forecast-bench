"""Validate src/data/clearsky_power.py against all three arrays.

Fits temperature climatology and gain on 2012-2013 only, then prints gain
diagnostics and the clear-sky power index k_p by hour of day and by month
over training daylight hours. No plotting, no files written.

Usage:
    python scripts/validate_clearsky_power.py
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
    TRAIN_YEARS,
    add_clearsky_power,
    fit_gain,
    fit_temperature_climatology,
    model_clearsky_power,
)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# name, parquet filename, nameplate kW, gamma_pdc
ARRAYS = [
    ("array07_CdTe", "array07_CdTe_hourly.parquet", 7.0, GAMMA_PDC_CDTE),
    ("array11_polySi", "array11_polySi_hourly.parquet", 5.0, GAMMA_PDC_SILICON),
    ("array12_monoSi", "array12_monoSi_hourly.parquet", 5.1, GAMMA_PDC_SILICON),
]


def main() -> None:
    for name, filename, nameplate_kw, gamma_pdc in ARRAYS:
        print(f"\n{'=' * 60}")
        print(f"{name} (nameplate {nameplate_kw} kW, gamma_pdc {gamma_pdc})")
        print("=" * 60)

        df = pd.read_parquet(PROCESSED_DIR / filename)
        df = add_solar_position(df)
        df = add_clearsky(df)
        df = add_daylight_mask(df)
        df = add_clearsky_index_ghi(df)

        train_df = df[df.index.year.isin(TRAIN_YEARS)]

        temp_clim = fit_temperature_climatology(train_df)
        p_cs_raw_train = model_clearsky_power(
            train_df.index, nameplate_kw, gamma_pdc, temp_clim
        )
        gain, n_hours, iqr = fit_gain(train_df, p_cs_raw_train)

        train_df = add_clearsky_power(train_df, p_cs_raw_train, gain, nameplate_kw)

        # 1. gain, hours used, IQR
        print(f"\nGain: {gain:.4f}  (hours used: {n_hours}, IQR: {iqr:.4f})")

        daylight = train_df[train_df["is_daylight"]]

        # 2. k_p percentiles over training daylight hours
        print("\nk_p percentiles (training daylight hours):")
        qs = daylight["k_p"].quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
        for q, val in qs.items():
            print(f"  p{int(round(q * 100))}: {val:.3f}")

        # 3. mean k_p by hour of day
        print("\nMean k_p by hour of day (training daylight hours):")
        by_hour = daylight.groupby(daylight.index.hour)["k_p"].mean()
        by_hour.index.name = "hour"
        print(by_hour.round(3).to_string())

        # 4. mean k_p by month
        print("\nMean k_p by month (training daylight hours):")
        by_month = daylight.groupby(daylight.index.month)["k_p"].mean()
        by_month.index.name = "month"
        print(by_month.round(3).to_string())


if __name__ == "__main__":
    main()
