"""Validate src/features/build.py on array11 for horizons 1, 3, 6.

Uses train+val (2009-2014) only, built the same way as
scripts/validate_persistence.py: clear-sky power's temperature climatology
and gain are fit on TRAINING years only, then applied to train+val's own
index. The 2015 test split is never loaded here - this script is feature-
building development/validation, not a modelling result, and per
CLAUDE.md the test set is touched once, at the end.

Prints, per horizon:
1. number of features and number of rows, per regime
2. assert_no_leakage PASS/FAIL, per regime
3. the five features most correlated with y, for the lagged regime, with a
   warning if any exceeds 0.99 correlation (a leakage red flag)
4. for horizon 1, lagged regime: the first 3 rows with their target-time
   index, to eyeball the alignment by hand

Usage:
    python scripts/validate_features.py
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
from src.features.build import assert_no_leakage, build_features, feature_names

PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "array11_polySi_hourly.parquet"
NAMEPLATE_KW = 5.0
GAMMA_PDC = GAMMA_PDC_SILICON

HORIZONS = [1, 3, 6]
REGIMES = ["lagged", "oracle"]


def load_array11_train_val():
    df = pd.read_parquet(PROCESSED_PATH)
    df = add_solar_position(df)
    df = add_clearsky(df)
    df = add_daylight_mask(df)
    df = add_clearsky_index_ghi(df)

    train, val, _test = split_chronological(df)  # 2015 test never touched

    temp_clim = fit_temperature_climatology(train)
    p_cs_raw_train = model_clearsky_power(train.index, NAMEPLATE_KW, GAMMA_PDC, temp_clim)
    gain, n_gain_hours, gain_iqr = fit_gain(train, p_cs_raw_train)
    print(f"gain (fit on training years only): {gain:.4f}  "
          f"(hours used: {n_gain_hours}, IQR: {gain_iqr:.4f})")

    train_val = pd.concat([train, val]).sort_index()
    p_cs_raw = model_clearsky_power(train_val.index, NAMEPLATE_KW, GAMMA_PDC, temp_clim)
    train_val = add_clearsky_power(train_val, p_cs_raw, gain, NAMEPLATE_KW)

    return train_val


def print_top_correlations(X, y, n=5):
    corr = X.corrwith(y).abs().sort_values(ascending=False)
    top = corr.head(n)
    print(f"  top {n} |correlation| with y:")
    flagged = False
    for name, value in top.items():
        flag = "  <-- RED FLAG (> 0.99)" if value > 0.99 else ""
        if value > 0.99:
            flagged = True
        print(f"    {name:35s} {value:.4f}{flag}")
    if flagged:
        print("  WARNING: at least one feature correlates above 0.99 with y - "
              "check for leakage before trusting this regime.")


def main():
    df = load_array11_train_val()

    for horizon in HORIZONS:
        print(f"\n{'=' * 78}")
        print(f"array11_polySi, horizon = {horizon}h")
        print("=" * 78)

        built = {}
        for regime in REGIMES:
            X, y = build_features(df, horizon, regime)
            built[regime] = (X, y)

            n_features = len(feature_names(regime, horizon))
            print(f"\n[{regime}] n_features={n_features}  n_rows={len(X)}")

            try:
                assert_no_leakage(df, X, y, horizon, regime)
                print(f"[{regime}] assert_no_leakage: PASS")
            except AssertionError as e:
                print(f"[{regime}] assert_no_leakage: FAIL - {e}")

        X_lagged, y_lagged = built["lagged"]
        print(f"\n[lagged] leakage probe - correlation of each feature with y:")
        print_top_correlations(X_lagged, y_lagged)

        if horizon == 1:
            print(f"\n[lagged] first 3 rows (horizon=1), target-time index:")
            preview_cols = [
                "Active_Power_issue",
                "k_p_issue",
                "p_cs",
                "hour_sin",
                "hour_cos",
            ]
            preview = X_lagged[preview_cols].head(3).copy()
            preview["y"] = y_lagged.head(3)
            with pd.option_context("display.width", 160, "display.max_columns", 20):
                print(preview)


if __name__ == "__main__":
    main()
