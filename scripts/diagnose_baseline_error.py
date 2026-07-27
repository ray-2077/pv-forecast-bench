"""Diagnose smart persistence's error: genuine forecasting difficulty versus
a systematic diurnal bias in k_p from clear-sky model error at low solar
elevation.

Every skill score in the paper is measured against SmartPersistence
(CLAUDE.md rule 4), so this decides whether those skill scores are
trustworthy or partly an artefact of a baseline that is handicapped at
specific hours rather than genuinely hard to beat there.

array11, VALIDATION split (2014) ONLY - 2015 is never loaded past
split_chronological's mechanical partitioning, and its rows are never read.

Diagnosis only: no fixes, no plotting, no changes to persistence.py or
clearsky_power.py - both are only imported and called.

Usage:
    python scripts/diagnose_baseline_error.py
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
from src.eval.metrics import mbe, nrmse, rmse
from src.models.base import check_no_lookahead
from src.models.persistence import SmartPersistence

PROCESSED_DIR = REPO_ROOT / "data" / "processed"

ARRAY_FILE = "array11_polySi_hourly.parquet"
NAMEPLATE_KW = 5.0
GAMMA_PDC = GAMMA_PDC_SILICON

HORIZONS = [1, 3, 6]

# The main per-horizon pass (items 1-3) uses this threshold, matching
# src.data.clearsky.add_daylight_mask's own default. Item 4 sweeps all
# three, including this one again, to see whether a stricter filter
# removes the systematic component.
DEFAULT_ELEVATION_THRESHOLD = 10.0
ELEVATION_THRESHOLDS = (10.0, 15.0, 20.0)


def load_val_split():
    """Load array11, build every column needed, and split chronologically.
    The test (2015) partition is produced by split_chronological's
    bookkeeping but is never assigned past this function - not read, not
    printed, not touched.
    """
    df = pd.read_parquet(PROCESSED_DIR / ARRAY_FILE)
    df = add_solar_position(df)
    df = add_clearsky(df)
    df = add_daylight_mask(df)
    df = add_clearsky_index_ghi(df)

    train, val, _test = split_chronological(df)

    temp_clim = fit_temperature_climatology(train)
    p_cs_raw_train = model_clearsky_power(train.index, NAMEPLATE_KW, GAMMA_PDC, temp_clim)
    gain, n_gain_hours, gain_iqr = fit_gain(train, p_cs_raw_train)
    train = add_clearsky_power(train, p_cs_raw_train, gain, NAMEPLATE_KW)

    p_cs_raw_val = model_clearsky_power(val.index, NAMEPLATE_KW, GAMMA_PDC, temp_clim)
    val = add_clearsky_power(val, p_cs_raw_val, gain, NAMEPLATE_KW)

    print(
        f"gain (fit on train years only): {gain:.4f}  "
        f"(hours used: {n_gain_hours}, IQR: {gain_iqr:.4f})"
    )

    return train, val


def smart_persistence_predictions(train, val, horizon):
    """SmartPersistence predictions and signed error (pred - actual) on
    val, indexed by target time, per the base class convention.
    """
    model = SmartPersistence()
    model.fit(train, horizon)  # no-op, see persistence.py
    preds = model.predict(val, horizon)
    check_no_lookahead(val, preds, horizon)

    y_true = val.loc[preds.index, "Active_Power"]
    error = preds - y_true
    return preds, y_true, error


def k_p_change_frame(val, horizon):
    """Per-timestamp k_p, its value h hours earlier (raw, not
    forward-filled - this measures genuine k_p variability, not whatever
    SmartPersistence's own fallback smooths over), and the signed/absolute
    change between them. NaN wherever either side is undefined (night, or
    a gap), which pandas' groupby().mean()/.std() skip automatically.
    """
    out = pd.DataFrame(index=val.index)
    out["hour"] = val.index.hour
    out["k_p"] = val["k_p"]
    out["k_p_lag"] = val["k_p"].shift(horizon)
    out["change"] = out["k_p"] - out["k_p_lag"]
    out["abs_change"] = out["change"].abs()
    return out


def error_by_hour(hour, error, mask):
    """n, MBE, RMSE of `error` (pred - actual) grouped by hour of day,
    restricted to `mask` (boolean, aligned with hour/error). Rows where a
    given hour has zero samples under mask are omitted.
    """
    sub = pd.DataFrame({"hour": hour[mask], "error": error[mask]})
    rows = []
    for h, group in sub.groupby("hour"):
        e = group["error"].to_numpy()
        rows.append(
            {
                "hour": int(h),
                "n": len(e),
                "mbe": float(e.mean()),
                "rmse": float((e ** 2).mean() ** 0.5),
            }
        )
    return pd.DataFrame(rows).sort_values("hour")


def change_decomposition_by_hour(diag, mask):
    """n, mean signed change, std of change, and |mean|/std for the k_p
    change series, grouped by hour of day, restricted to `mask`.
    """
    sub = diag.loc[mask, ["hour", "change"]].dropna()
    rows = []
    for h, group in sub.groupby("hour"):
        c = group["change"].to_numpy()
        n = len(c)
        mean = float(c.mean())
        std = float(c.std(ddof=1)) if n > 1 else float("nan")
        ratio = abs(mean) / std if n > 1 and std > 0 else float("nan")
        rows.append({"hour": int(h), "n": n, "mean_change": mean, "std_change": std, "abs_mean_over_std": ratio})
    return pd.DataFrame(rows).sort_values("hour")


def mean_k_p_by_hour(diag, mask):
    """n, mean k_p, mean |k_p(t) - k_p(t-h)| grouped by hour of day."""
    sub = diag.loc[mask, ["hour", "k_p", "abs_change"]]
    rows = []
    for h, group in sub.groupby("hour"):
        k_p_vals = group["k_p"].dropna()
        abs_change_vals = group["abs_change"].dropna()
        rows.append(
            {
                "hour": int(h),
                "n_k_p": len(k_p_vals),
                "mean_k_p": float(k_p_vals.mean()) if len(k_p_vals) else float("nan"),
                "n_change": len(abs_change_vals),
                "mean_abs_change": float(abs_change_vals.mean()) if len(abs_change_vals) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("hour")


def print_error_table(df):
    for _, r in df.iterrows():
        print(f"    hour {r['hour']:02.0f}  n={r['n']:5.0f}  MBE={r['mbe']:+.4f} kW  RMSE={r['rmse']:.4f} kW")


def print_kp_table(df):
    for _, r in df.iterrows():
        print(
            f"    hour {r['hour']:02.0f}  n={r['n_k_p']:5.0f}  mean k_p={r['mean_k_p']:+.3f}  "
            f"n(change)={r['n_change']:5.0f}  mean|delta k_p|={r['mean_abs_change']:.3f}"
        )


def print_decomp_table(df):
    for _, r in df.iterrows():
        ratio = r["abs_mean_over_std"]
        ratio_str = f"{ratio:.2f}" if ratio == ratio else "  nan"  # NaN check
        print(
            f"    hour {r['hour']:02.0f}  n={r['n']:5.0f}  mean delta k_p={r['mean_change']:+.4f}  "
            f"std delta k_p={r['std_change']:.4f}  |mean|/std={ratio_str}"
        )


def main():
    train, val = load_val_split()

    for horizon in HORIZONS:
        print(f"\n{'=' * 78}")
        print(f"horizon = {horizon}h  (array11, validation split 2014)")
        print("=" * 78)

        preds, y_true, error = smart_persistence_predictions(train, val, horizon)
        hour = preds.index.hour
        elevation = val.loc[preds.index, "solar_elevation"]

        diag = k_p_change_frame(val, horizon)
        # Restrict diag to the same target timestamps SmartPersistence
        # actually produced a prediction for, so items 1-3 describe the
        # same set of rows throughout.
        diag = diag.loc[preds.index]

        default_mask = (elevation > DEFAULT_ELEVATION_THRESHOLD).to_numpy()

        print(f"\n--- 1. smart persistence error by hour of day (elevation > {DEFAULT_ELEVATION_THRESHOLD:.0f} deg) ---")
        print_error_table(error_by_hour(hour, error.to_numpy(), default_mask))

        print(f"\n--- 2. mean k_p and mean |k_p(t) - k_p(t-{horizon})| by hour of day (elevation > {DEFAULT_ELEVATION_THRESHOLD:.0f} deg) ---")
        print_kp_table(mean_k_p_by_hour(diag, default_mask))

        print(f"\n--- 3. decomposition of k_p(t) - k_p(t-{horizon}): mean (systematic) vs std (variability), elevation > {DEFAULT_ELEVATION_THRESHOLD:.0f} deg ---")
        print_decomp_table(change_decomposition_by_hour(diag, default_mask))

        print(f"\n--- 4. threshold sweep: repeat 1 and 3, plus aggregate daylight RMSE/nRMSE ---")
        for threshold in ELEVATION_THRESHOLDS:
            mask = (elevation > threshold).to_numpy()
            n = int(mask.sum())
            if n == 0:
                print(f"\n  [elevation > {threshold:.0f} deg] n=0, skipping")
                continue
            agg_rmse = rmse(y_true.to_numpy()[mask], preds.to_numpy()[mask])
            agg_nrmse = nrmse(y_true.to_numpy()[mask], preds.to_numpy()[mask], NAMEPLATE_KW)
            agg_mbe = mbe(y_true.to_numpy()[mask], preds.to_numpy()[mask])
            print(f"\n  [elevation > {threshold:.0f} deg]  n={n}  daylight RMSE={agg_rmse:.4f} kW  "
                  f"nRMSE={agg_nrmse:.2f}%  MBE={agg_mbe:+.4f} kW")
            print("  repeat of 1:")
            print_error_table(error_by_hour(hour, error.to_numpy(), mask))
            print("  repeat of 3:")
            print_decomp_table(change_decomposition_by_hour(diag, mask))


if __name__ == "__main__":
    main()
