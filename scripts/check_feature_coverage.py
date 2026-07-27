"""Diagnose whether build_features' row-dropping (src/features/build.py)
removes targets that smart persistence (src/models/persistence.py) can
still forecast.

Why this matters: smart persistence only needs k_p at t-horizon (with up
to 24h forward-fill) to predict t. build_features' lagged regime requires
a full stack of lag/rolling features to all be non-NaN, which is a much
stricter requirement. If that drops a different, smaller set of daylight
targets than smart persistence would fail to predict, then any skill-score
comparison between smart persistence and a model trained on build_features'
output is being computed over two different sample sets - not a fair
comparison. This script only measures the mismatch; it does not fix it.

array11 only, train+val (2009-2014). The 2015 test split is never loaded.

No fixes, no plotting, no file writing - diagnosis only.

Usage:
    python scripts/check_feature_coverage.py
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
    GAMMA_PDC_SILICON,
    add_clearsky_power,
    fit_gain,
    fit_temperature_climatology,
    model_clearsky_power,
)
from src.data.splits import split_chronological
from src.features.build import build_features

PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "array11_polySi_hourly.parquet"
NAMEPLATE_KW = 5.0
GAMMA_PDC = GAMMA_PDC_SILICON

HORIZONS = [1, 3, 6]
REGIME = "lagged"


def load_array11_train_val():
    df = pd.read_parquet(PROCESSED_PATH)
    df = add_solar_position(df)
    df = add_clearsky(df)
    df = add_daylight_mask(df)
    df = add_clearsky_index_ghi(df)

    train, val, _test = split_chronological(df)  # 2015 test never touched

    temp_clim = fit_temperature_climatology(train)
    p_cs_raw_train = model_clearsky_power(train.index, NAMEPLATE_KW, GAMMA_PDC, temp_clim)
    gain, _n_gain_hours, _gain_iqr = fit_gain(train, p_cs_raw_train)

    train_val = pd.concat([train, val]).sort_index()
    p_cs_raw = model_clearsky_power(train_val.index, NAMEPLATE_KW, GAMMA_PDC, temp_clim)
    train_val = add_clearsky_power(train_val, p_cs_raw, gain, NAMEPLATE_KW)

    return train_val


def issue_time_is_daylight(df, horizon):
    """For each target time t in df.index, the is_daylight value at the
    issue time t - horizon hours, or NaN if t - horizon falls before the
    start of df.
    """
    issue_times = df.index - pd.Timedelta(hours=horizon)
    looked_up = df["is_daylight"].reindex(issue_times)
    return pd.Series(looked_up.to_numpy(), index=df.index)


def coverage_row(df, horizon):
    X, _y = build_features(df, horizon, REGIME)
    survived_idx = X.index

    total_rows = len(df)
    survived_rows = len(survived_idx)

    daylight_idx = df.index[df["is_daylight"]]
    daylight_total = len(daylight_idx)
    daylight_survived_idx = daylight_idx.intersection(survived_idx)
    daylight_survived = len(daylight_survived_idx)
    daylight_retention_pct = (
        100.0 * daylight_survived / daylight_total if daylight_total else float("nan")
    )

    daylight_dropped_idx = daylight_idx.difference(survived_idx)
    issue_daylight = issue_time_is_daylight(df, horizon)
    dropped_with_night_issue = int((issue_daylight.loc[daylight_dropped_idx] == False).sum())

    night_idx = df.index[~df["is_daylight"]]
    night_survived = len(night_idx.intersection(survived_idx))

    return {
        "horizon": horizon,
        "total_rows": total_rows,
        "survived_rows": survived_rows,
        "daylight_total": daylight_total,
        "daylight_survived": daylight_survived,
        "daylight_retention_pct": daylight_retention_pct,
        "daylight_dropped": len(daylight_dropped_idx),
        "daylight_dropped_night_issue": dropped_with_night_issue,
        "night_survived": night_survived,
        "daylight_dropped_idx": daylight_dropped_idx,
    }


def print_table(rows):
    header = (
        f"{'h':>2} {'total':>7} {'survived':>9} {'daylight_tot':>13} "
        f"{'daylight_surv':>14} {'retain%':>8} {'dropped':>8} "
        f"{'drop_night_issue':>17} {'night_surv':>11}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['horizon']:>2} {r['total_rows']:>7} {r['survived_rows']:>9} "
            f"{r['daylight_total']:>13} {r['daylight_survived']:>14} "
            f"{r['daylight_retention_pct']:>7.2f}% {r['daylight_dropped']:>8} "
            f"{r['daylight_dropped_night_issue']:>17} {r['night_survived']:>11}"
        )


def print_hourly_breakdown(dropped_idx, horizon):
    print(f"\nhorizon={horizon}h: dropped daylight targets by hour of day")
    if len(dropped_idx) == 0:
        print("  none dropped")
        return
    counts = pd.Series(dropped_idx.hour).value_counts().sort_index()
    for hour, count in counts.items():
        print(f"  hour {hour:02d}: {count}")


def main():
    df = load_array11_train_val()

    rows = [coverage_row(df, h) for h in HORIZONS]
    print_table(rows)

    h6 = next(r for r in rows if r["horizon"] == 6)
    print_hourly_breakdown(h6["daylight_dropped_idx"], 6)


if __name__ == "__main__":
    main()
