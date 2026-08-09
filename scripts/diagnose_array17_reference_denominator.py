"""Diagnose the array17 val-to-test skill_vs_convex shift: final check,
the reference forecasts themselves (skill is a RATIO - a stronger
denominator lowers skill with no change in the model's own accuracy).

CONTEXT (see scripts/diagnose_array17_test_shift.py and
scripts/diagnose_array17_target_volatility.py for the other four checks,
all inconclusive or contradicted by the data): array17's test
skill_vs_convex is lower than val in all 15 model x horizon cells;
array11 and array12 both move the other way in all 18 of their cells.

This script prints, per array, per horizon, for val and test, daylight
hours, outage-excluded:
  - RMSE and nRMSE (normalised by nameplate_kw) of smart persistence
  - RMSE and nRMSE of climatology
  - RMSE and nRMSE of the convex reference (the actual skill_vs_convex
    denominator)
  - each model's own RMSE on both splits (mean over 5 seeds, from the
    already-committed results/seed_sweep_summary_lagged[_test].csv - not
    refit here), so numerator and denominator can be read side by side

If array17's convex-reference RMSE fell from val to test while array11
and array12's rose, the shift is a denominator effect: the reference got
harder to beat for array17 and easier for the other two, which lowers
skill_vs_convex with no change in how good the model itself is. Smart
persistence and climatology are printed too because the convex
reference is a blend of exactly those two - if one of them moved and
not the other, that is itself informative about which mechanism (a
better recent observation to persist from, or a better climatological
average) is doing the work.

The convex weight w is fit on VAL ONLY in every reference computed here,
matching every real run (CLAUDE.md rule 4) - both the val-evaluated and
the test-evaluated convex reference use the SAME val-fit w; only the
split being PREDICTED changes. This is deliberately different from
scripts/diagnose_array17_test_shift.py's separate diagnostic-only
test-fit w (which existed purely to ask whether the optimal w differs by
year); this script always uses the real, reported w.

Uses the same pipeline as the other two diagnose_array17_*.py scripts:
load_and_prepare -> split_chronological -> temperature climatology and
gain fit on TRAIN ONLY -> clear-sky power applied unchanged to val and
to test (add_clearsky_power_to_test, duplicated locally again - see
either sibling script for why this is not imported from
scripts/run_final_test.py, which is write-once).

No fixes, no plotting, no changes to src/. Diagnosis only.

Usage:
    python scripts/diagnose_array17_reference_denominator.py
"""

import csv
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
from src.eval.metrics import nrmse, rmse
from src.models.climatology import ConvexCombination, Climatology
from src.models.persistence import SmartPersistence

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

ARRAYS = ["array11", "array12", "array17"]
HORIZONS = (1, 3, 6)
MODELS = ["xgboost", "lstm", "cnn_lstm", "lstm_residual", "cnn_lstm_residual"]


def add_clearsky_power_to_test(train, test, nameplate_kw, gamma_pdc):
    """Duplicated from the sibling diagnose_array17_*.py scripts - see
    either for the rationale (train-only fit applied unchanged to test;
    not imported from scripts/run_final_test.py, which is write-once).
    """
    temp_clim = fit_temperature_climatology(train)
    p_cs_raw_train = model_clearsky_power(train.index, nameplate_kw, gamma_pdc, temp_clim)
    gain, _n_gain_hours, _gain_iqr = fit_gain(train, p_cs_raw_train)

    p_cs_raw_test = model_clearsky_power(test.index, nameplate_kw, gamma_pdc, temp_clim)
    return add_clearsky_power(test, p_cs_raw_test, gain, nameplate_kw)


def load_splits(array):
    df, nameplate_kw, gamma_pdc = load_and_prepare(array, PROCESSED_DIR)
    train, val, test = split_chronological(df)
    train, val, _gain_info = add_clearsky_power_per_split(train, val, test, nameplate_kw, gamma_pdc)
    test = add_clearsky_power_to_test(train, test, nameplate_kw, gamma_pdc)
    return train, val, test, nameplate_kw


def eval_index(array, split_df, common_idx):
    """common_idx restricted to daylight and outage-excluded - the same
    population every real run (scripts/build_table6_dm.py,
    scripts/run_final_test.py) evaluates on.
    """
    daylight_idx = common_idx[split_df.loc[common_idx, "is_daylight"].to_numpy()]
    outage = exclusion_mask(array, daylight_idx)
    return daylight_idx[~outage.to_numpy()]


def reference_metrics_for_horizon(array, train, val, test, horizon, nameplate_kw):
    """{split_label: {"persistence": (rmse, nrmse), "climatology": (...),
    "convex": (...), "n_eval": n}} for val and test, this (array, horizon).
    convex weight w is fit on VAL ONLY for both splits - see module
    docstring.
    """
    sp_model = SmartPersistence().fit(train, horizon)
    clim_model = Climatology().fit(train, horizon)
    convex_model = ConvexCombination().fit(train, horizon, val)  # w fit on VAL ONLY, always

    out = {}
    for split_label, split_df in [("val", val), ("test", test)]:
        preds_sp = sp_model.predict(split_df, horizon)
        preds_clim = clim_model.predict(split_df, horizon)
        preds_convex = convex_model.predict(split_df, horizon)

        common_idx = preds_sp.index.intersection(preds_clim.index).intersection(preds_convex.index)
        eval_idx = eval_index(array, split_df, common_idx)

        y_true = split_df.loc[eval_idx, "Active_Power"]
        out[split_label] = {
            "n_eval": len(eval_idx),
            "convex_weight": convex_model.w,
            "rmse_persistence": rmse(y_true, preds_sp.loc[eval_idx]),
            "nrmse_persistence": nrmse(y_true, preds_sp.loc[eval_idx], nameplate_kw),
            "rmse_climatology": rmse(y_true, preds_clim.loc[eval_idx]),
            "nrmse_climatology": nrmse(y_true, preds_clim.loc[eval_idx], nameplate_kw),
            "rmse_convex": rmse(y_true, preds_convex.loc[eval_idx]),
            "nrmse_convex": nrmse(y_true, preds_convex.loc[eval_idx], nameplate_kw),
        }
    return out


def load_model_rmse():
    """{(model, array, horizon): {"val": mean_rmse_daylight, "test": ...}}
    from the already-committed seed_sweep_summary_lagged[_test].csv - not
    refit here (5-seed means, same population every real run evaluates
    on: daylight, outage-excluded).
    """
    out = {}
    for split_label, filename in [("val", "seed_sweep_summary_lagged.csv"), ("test", "seed_sweep_summary_lagged_test.csv")]:
        with open(RESULTS_DIR / filename, newline="") as fh:
            for row in csv.DictReader(fh):
                key = (row["model"], row["array"], int(row["horizon"]))
                out.setdefault(key, {})[split_label] = float(row["mean_rmse_daylight"])
    return out


def main():
    print("=" * 100)
    print("FINAL CHECK: reference-forecast (denominator) strength, val (2014) vs test (2015)")
    print("daylight hours, outage-excluded - same population every real run evaluates on")
    print("=" * 100)

    model_rmse = load_model_rmse()

    ref_rows = []
    for array in ARRAYS:
        train, val, test, nameplate_kw = load_splits(array)
        for horizon in HORIZONS:
            metrics = reference_metrics_for_horizon(array, train, val, test, horizon, nameplate_kw)
            for split_label in ("val", "test"):
                m = metrics[split_label]
                ref_rows.append(
                    {
                        "array": array,
                        "horizon": horizon,
                        "split": split_label,
                        "n_eval": m["n_eval"],
                        "convex_weight": m["convex_weight"],
                        "rmse_persistence": m["rmse_persistence"],
                        "nrmse_persistence": m["nrmse_persistence"],
                        "rmse_climatology": m["rmse_climatology"],
                        "nrmse_climatology": m["nrmse_climatology"],
                        "rmse_convex": m["rmse_convex"],
                        "nrmse_convex": m["nrmse_convex"],
                    }
                )

    ref_df = pd.DataFrame(ref_rows).set_index(["array", "horizon", "split"])

    print("\n--- reference forecast RMSE / nRMSE(%), by array x horizon x split ---\n")
    print(ref_df.round(4).to_string())

    print("\n--- val -> test change in the CONVEX reference (denominator), per array x horizon ---")
    print("(negative d_rmse_convex = reference got STRONGER/harder to beat on test = pulls skill DOWN)\n")
    delta_rows = []
    for array in ARRAYS:
        for horizon in HORIZONS:
            v = ref_df.loc[(array, horizon, "val")]
            t = ref_df.loc[(array, horizon, "test")]
            delta_rows.append(
                {
                    "array": array,
                    "horizon": horizon,
                    "d_rmse_persistence": t["rmse_persistence"] - v["rmse_persistence"],
                    "d_rmse_climatology": t["rmse_climatology"] - v["rmse_climatology"],
                    "d_rmse_convex": t["rmse_convex"] - v["rmse_convex"],
                }
            )
    delta_df = pd.DataFrame(delta_rows).set_index(["array", "horizon"])
    print(delta_df.round(4).to_string())

    print("\n--- numerator vs denominator: each model's own RMSE (5-seed mean) vs the convex-reference RMSE ---\n")
    combo_rows = []
    for array in ARRAYS:
        for horizon in HORIZONS:
            ref_v = ref_df.loc[(array, horizon, "val"), "rmse_convex"]
            ref_t = ref_df.loc[(array, horizon, "test"), "rmse_convex"]
            for model in MODELS:
                mv = model_rmse[(model, array, horizon)]["val"]
                mt = model_rmse[(model, array, horizon)]["test"]
                combo_rows.append(
                    {
                        "array": array,
                        "horizon": horizon,
                        "model": model,
                        "model_rmse_val": mv,
                        "model_rmse_test": mt,
                        "d_model_rmse": mt - mv,
                        "convex_rmse_val": ref_v,
                        "convex_rmse_test": ref_t,
                        "d_convex_rmse": ref_t - ref_v,
                        "skill_val_recomputed": 1 - mv / ref_v,
                        "skill_test_recomputed": 1 - mt / ref_t,
                    }
                )
    combo_df = pd.DataFrame(combo_rows).set_index(["array", "horizon", "model"])
    print(combo_df.round(4).to_string())


if __name__ == "__main__":
    main()
