"""Load raw DKASC array CSVs into clean hourly DataFrames.

Pipeline: load_array -> clean_5min -> resample_hourly. No scalers, no
normalisation, no train/test logic here - that happens downstream once the
window is split.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Australia/Darwin"

KEEP_COLS = [
    "timestamp",
    "Active_Power",
    "Global_Horizontal_Radiation",
    "Diffuse_Horizontal_Radiation",
    "Weather_Temperature_Celsius",
    "Weather_Relative_Humidity",
    "Wind_Speed",
]

# Dropped from the raw CSVs:
# - Active_Energy_Delivered_Received, Performance_Ratio, Current_Phase_Average:
#   all derived from Active_Power (or power + irradiance) and leak the target.
# - Wind_Direction, Weather_Daily_Rainfall, Radiation_Global_Tilted,
#   Radiation_Diffuse_Tilted: not used in this analysis.

# Physical plausibility bounds. Active_Power's upper bound depends on
# nameplate_kw and is built inside clean_5min.
RANGE_BOUNDS = {
    "Global_Horizontal_Radiation": (-10, 1400),
    "Diffuse_Horizontal_Radiation": (-10, 1400),
    "Weather_Temperature_Celsius": (-10, 55),
    "Weather_Relative_Humidity": (0, 100),
    "Wind_Speed": (0, 40),
}

CLIP_ZERO_COLS = [
    "Active_Power",
    "Global_Horizontal_Radiation",
    "Diffuse_Horizontal_Radiation",
]

SLOTS_PER_HOUR = 12  # 5-minute samples per hour


def load_array(path, start="2012-01-01", end="2015-12-31"):
    """Read one array's raw CSV, keep only the non-leaking columns.

    Timestamps are naive ACST 5-minute samples; NT has no daylight saving
    so tz_localize is unambiguous. Returns a tz-aware DataFrame indexed by
    timestamp at native 5-minute resolution, restricted to [start, end]
    inclusive of both endpoints.
    """
    path = Path(path)
    df = pd.read_csv(path, usecols=KEEP_COLS)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    df.index = df.index.tz_localize(TZ)

    start_ts = pd.Timestamp(start, tz=TZ)
    end_ts = pd.Timestamp(end, tz=TZ) + pd.Timedelta(days=1)
    df = df.loc[(df.index >= start_ts) & (df.index < end_ts)]

    return df


def clean_5min(df, nameplate_kw):
    """Physical plausibility filtering at native 5-minute resolution.

    Out-of-range values become NaN (no interpolation, no fill). Then
    negative Active_Power and negative radiation are clipped to 0. Returns
    the cleaned frame and a dict of {column: n_values_set_to_nan}.
    """
    df = df.copy()
    removed = {}

    bounds = dict(RANGE_BOUNDS)
    bounds["Active_Power"] = (-0.1, 1.5 * nameplate_kw)

    for col, (lo, hi) in bounds.items():
        if col not in df.columns:
            continue
        out_of_range = (df[col] < lo) | (df[col] > hi)
        removed[col] = int(out_of_range.sum())
        df.loc[out_of_range, col] = np.nan

    for col in CLIP_ZERO_COLS:
        df[col] = df[col].clip(lower=0)

    return df, removed


def resample_hourly(df, min_samples=9):
    """Resample to hourly means, hour-beginning labels.

    The 12:00 row covers 12:00-12:55. An hour's value for a column is NaN
    if fewer than min_samples (out of 12) five-minute values were non-NaN
    for that column. No gap filling.
    """
    resampler = df.resample("1h", closed="left", label="left")
    means = resampler.mean()
    counts = resampler.count()
    return means.where(counts >= min_samples)
