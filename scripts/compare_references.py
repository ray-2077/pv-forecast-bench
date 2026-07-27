"""Compare the three reference forecasters - SmartPersistence, Climatology,
ConvexCombination (src/models/climatology.py) - on the VALIDATION split
(2014) ONLY, for arrays 11, 12, 17. 2015 is never loaded past
split_chronological's mechanical partitioning, and its rows are never read.
(array07 excluded - see CLAUDE.md "Data window" and results/dead_period_audit.csv.)

Motivation: scripts/diagnose_baseline_error.py showed SmartPersistence has
a large, systematic MBE at midday for longer horizons (issue time near
dawn -> stale forward-filled k_p). Every skill score reported for XGBoost
elsewhere in this project is measured against SmartPersistence, so if that
baseline is structurally broken at specific hours, those skill scores are
inflated relative to what a defensible reference (Yang et al. 2020) would
show. This script quantifies the difference, and checks whether the
collapse is specific to array11 or shared across the three co-located
arrays (they share one weather station - CLAUDE.md's data-window notes).

For "the existing XGBoost run": only h=3 (lagged, array11, seed 0) has a
results/*.json on disk. To report a real (not fabricated) skill number at
every horizon and array, this script fits XGBForecaster live, with the
exact same default hyperparameters and seed as that on-disk run - i.e. it
reproduces the existing model/config, not a new one - for all three
horizons and all three arrays. This is a real computation each time, never
a placeholder, per CLAUDE.md's research-integrity rules.

Writes two CSVs (in addition to the console report):
  - results/reference_comparison.csv   one row per (array, horizon)
  - results/reference_mbe_by_hour.csv  one row per (array, horizon, hour),
    for the midday-collapse figure

No changes to persistence.py or any model code.

Usage:
    python scripts/compare_references.py
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
    GAMMA_PDC_HIT,
    GAMMA_PDC_SILICON,
    add_clearsky_power,
    fit_gain,
    fit_temperature_climatology,
    model_clearsky_power,
)
from src.data.splits import split_chronological
from src.eval.metrics import mbe, nrmse, rmse, skill_score
from src.models.base import check_no_lookahead
from src.models.climatology import Climatology, ConvexCombination
from src.models.persistence import SmartPersistence
from src.models.xgb import XGBForecaster

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

ARRAYS = {
    "array11": {"file": "array11_polySi_hourly.parquet", "nameplate_kw": 5.0, "gamma_pdc": GAMMA_PDC_SILICON},
    "array12": {"file": "array12_monoSi_hourly.parquet", "nameplate_kw": 5.1, "gamma_pdc": GAMMA_PDC_SILICON},
    "array17": {"file": "array17_HIT_hourly.parquet", "nameplate_kw": 6.3, "gamma_pdc": GAMMA_PDC_HIT},
}

HORIZONS = [1, 3, 6]
DAYLIGHT_ELEVATION_THRESHOLD = 10.0
XGB_SEED = 0
EVAL_SPLIT = "val"


def load_val_split(array_file, nameplate_kw, gamma_pdc):
    df = pd.read_parquet(PROCESSED_DIR / array_file)
    df = add_solar_position(df)
    df = add_clearsky(df)
    df = add_daylight_mask(df)
    df = add_clearsky_index_ghi(df)

    train, val, _test = split_chronological(df)

    temp_clim = fit_temperature_climatology(train)
    p_cs_raw_train = model_clearsky_power(train.index, nameplate_kw, gamma_pdc, temp_clim)
    gain, n_gain_hours, gain_iqr = fit_gain(train, p_cs_raw_train)
    train = add_clearsky_power(train, p_cs_raw_train, gain, nameplate_kw)

    p_cs_raw_val = model_clearsky_power(val.index, nameplate_kw, gamma_pdc, temp_clim)
    val = add_clearsky_power(val, p_cs_raw_val, gain, nameplate_kw)

    print(
        f"gain (fit on train years only): {gain:.4f}  "
        f"(hours used: {n_gain_hours}, IQR: {gain_iqr:.4f})"
    )
    return train, val


def daylight_subset(val, preds, threshold=DAYLIGHT_ELEVATION_THRESHOLD):
    """Restrict preds to rows where val's solar_elevation exceeds
    threshold, and return (y_true, y_pred) numpy arrays over that subset,
    aligned to preds' own valid index.
    """
    elevation = val.loc[preds.index, "solar_elevation"]
    mask = (elevation > threshold).to_numpy()
    y_true = val.loc[preds.index, "Active_Power"].to_numpy()[mask]
    y_pred = preds.to_numpy()[mask]
    return y_true, y_pred


def compute_daylight_metrics(val, preds, nameplate_kw):
    y_true, y_pred = daylight_subset(val, preds)
    return {
        "n": len(y_true),
        "rmse": rmse(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred, nameplate_kw),
        "mbe": mbe(y_true, y_pred),
    }


def print_daylight_metrics(label, m):
    print(
        f"  {label:16s}  n={m['n']:5d}  RMSE={m['rmse']:.4f} kW  "
        f"nRMSE={m['nrmse']:.2f}%  MBE={m['mbe']:+.4f} kW"
    )


def mbe_by_hour(val, preds, threshold=DAYLIGHT_ELEVATION_THRESHOLD):
    elevation = val.loc[preds.index, "solar_elevation"]
    mask = (elevation > threshold).to_numpy()
    y_true = val.loc[preds.index, "Active_Power"].to_numpy()
    y_pred = preds.to_numpy()
    hour = preds.index.hour.to_numpy()

    df = pd.DataFrame({"hour": hour[mask], "error": y_pred[mask] - y_true[mask]})
    rows = []
    for h, group in df.groupby("hour"):
        rows.append({"hour": int(h), "n": len(group), "mbe": float(group["error"].mean())})
    return pd.DataFrame(rows).sort_values("hour")


def print_mbe_by_hour(label, table):
    print(f"  {label}:")
    for _, r in table.iterrows():
        print(f"    hour {r['hour']:02.0f}  n={r['n']:4.0f}  MBE={r['mbe']:+.4f} kW")


def mbe_by_hour_rows(array, horizon, val, pers_preds, clim_preds, convex_preds):
    """One row per hour-of-day for this (array, horizon), columns
    array, horizon, hour, mbe_persistence, mbe_climatology, mbe_convex.
    Outer-merged on hour, since each forecaster's valid index (and
    therefore which daylight hours it has any rows for) can differ.
    """
    pers = mbe_by_hour(val, pers_preds)[["hour", "mbe"]].rename(columns={"mbe": "mbe_persistence"})
    clim = mbe_by_hour(val, clim_preds)[["hour", "mbe"]].rename(columns={"mbe": "mbe_climatology"})
    conv = mbe_by_hour(val, convex_preds)[["hour", "mbe"]].rename(columns={"mbe": "mbe_convex"})

    merged = pers.merge(clim, on="hour", how="outer").merge(conv, on="hour", how="outer")
    merged.insert(0, "horizon", horizon)
    merged.insert(0, "array", array)
    return merged.sort_values("hour").reset_index(drop=True)


def daylight_skill(val, preds_model, preds_ref, threshold=DAYLIGHT_ELEVATION_THRESHOLD):
    """Skill of preds_model vs preds_ref, over the intersection of their
    valid indices, restricted to daylight rows at the given threshold.
    """
    common_idx = preds_model.index.intersection(preds_ref.index)
    elevation = val.loc[common_idx, "solar_elevation"]
    mask = (elevation > threshold).to_numpy()

    y_true = val.loc[common_idx, "Active_Power"].to_numpy()[mask]
    y_model = preds_model.loc[common_idx].to_numpy()[mask]
    y_ref = preds_ref.loc[common_idx].to_numpy()[mask]

    return skill_score(y_true, y_model, y_ref), int(mask.sum())


def main():
    comparison_rows = []
    mbe_hour_rows = []

    for array, cfg in ARRAYS.items():
        nameplate_kw = cfg["nameplate_kw"]
        train, val = load_val_split(cfg["file"], nameplate_kw, cfg["gamma_pdc"])

        for horizon in HORIZONS:
            print(f"\n{'=' * 78}")
            print(f"array = {array}  horizon = {horizon}h  (validation split 2014)")
            print("=" * 78)

            persistence = SmartPersistence().fit(train, horizon)
            pers_preds = persistence.predict(val, horizon)
            check_no_lookahead(val, pers_preds, horizon)

            climatology = Climatology().fit(train, horizon)
            clim_preds = climatology.predict(val, horizon)

            convex = ConvexCombination().fit(train, horizon, val)
            convex_preds = convex.predict(val, horizon)
            check_no_lookahead(val, convex_preds, horizon)

            print(f"\nfitted convex combination weight w = {convex.w:.2f}  "
                  f"(w=1 -> pure persistence, w=0 -> pure climatology)")

            print("\n--- daylight RMSE / nRMSE / MBE, elevation > "
                  f"{DAYLIGHT_ELEVATION_THRESHOLD:.0f} deg ---")
            m_pers = compute_daylight_metrics(val, pers_preds, nameplate_kw)
            m_clim = compute_daylight_metrics(val, clim_preds, nameplate_kw)
            m_conv = compute_daylight_metrics(val, convex_preds, nameplate_kw)
            print_daylight_metrics("persistence", m_pers)
            print_daylight_metrics("climatology", m_clim)
            print_daylight_metrics("convex_reference", m_conv)

            hour_table = mbe_by_hour_rows(array, horizon, val, pers_preds, clim_preds, convex_preds)
            mbe_hour_rows.append(hour_table)

            if horizon == 6:
                print("\n--- MBE by hour of day at h=6 (does the blend fix the midday collapse?) ---")
                print_mbe_by_hour("persistence", mbe_by_hour(val, pers_preds))
                print_mbe_by_hour("climatology", mbe_by_hour(val, clim_preds))
                print_mbe_by_hour("convex_reference", mbe_by_hour(val, convex_preds))

            # --- XGBoost, same default config/seed as the on-disk h=3 run ---
            xgb_model = XGBForecaster(seed=XGB_SEED, regime="lagged")
            xgb_model.fit(train, horizon, df_val=val)
            xgb_preds = xgb_model.predict(val, horizon)
            check_no_lookahead(val, xgb_preds, horizon)

            skill_vs_persistence, n_vs_pers = daylight_skill(val, xgb_preds, pers_preds)
            skill_vs_convex, n_vs_convex = daylight_skill(val, xgb_preds, convex_preds)

            print(f"\n--- XGBoost (lagged, seed={XGB_SEED}) daylight skill score ---")
            print(f"  vs smart_persistence   n={n_vs_pers:5d}  skill={skill_vs_persistence:+.4f}")
            print(f"  vs convex_reference    n={n_vs_convex:5d}  skill={skill_vs_convex:+.4f}")

            # n_samples: the intersection of xgb and convex_reference's
            # valid daylight rows - the set the headline skill score
            # (rule 4, vs convex reference) is actually computed over.
            comparison_rows.append({
                "array": array,
                "horizon": horizon,
                "convex_weight": convex.w,
                "rmse_persistence": m_pers["rmse"],
                "rmse_climatology": m_clim["rmse"],
                "rmse_convex": m_conv["rmse"],
                "nrmse_persistence": m_pers["nrmse"],
                "nrmse_convex": m_conv["nrmse"],
                "mbe_persistence": m_pers["mbe"],
                "mbe_convex": m_conv["mbe"],
                "xgb_skill_vs_persistence": skill_vs_persistence,
                "xgb_skill_vs_convex": skill_vs_convex,
                "n_samples": n_vs_convex,
                "eval_split": EVAL_SPLIT,
            })

    RESULTS_DIR.mkdir(exist_ok=True)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = RESULTS_DIR / "reference_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nwrote {comparison_path}")

    mbe_hour_df = pd.concat(mbe_hour_rows, ignore_index=True)
    mbe_hour_path = RESULTS_DIR / "reference_mbe_by_hour.csv"
    mbe_hour_df.to_csv(mbe_hour_path, index=False)
    print(f"wrote {mbe_hour_path}")


if __name__ == "__main__":
    main()
