"""Development driver for XGBForecaster (src/models/xgb.py).

Evaluated on the VALIDATION split (2014) only, NOT the test split (2015) -
per CLAUDE.md's research-integrity rules, the test set is touched once, at
the end of the whole project. This script is model development, so it must
not look at 2015 results at all.

Pipeline: load one array's processed parquet -> solar position / clear-sky
irradiance / daylight mask / clear-sky index of GHI (src.data.clearsky) ->
chronological split (src.data.splits) -> temperature climatology and gain
fit on TRAIN only, then clear-sky POWER added to train and val
(src.data.clearsky_power) -> XGBForecaster.fit(train, horizon, df_val=val)
-> predict on val -> metrics against SmartPersistence AND the
ConvexCombination reference (src.models.climatology - weight fitted on
VALIDATION, per that module's own CRITICAL comment) on the same rows,
daylight-only and all-hours -> results/<run_id>.json.

Every run also records skill_vs_persistence, skill_vs_convex,
convex_weight, rmse_persistence, rmse_convex, rmse_climatology in each
metrics subset. Motivation: scripts/compare_references.py showed
XGBoost's apparent skill can differ enormously by which reference it is
measured against (h=6 on array11: +0.659 vs persistence, +0.215 vs the
convex combination) - a run JSON that only records skill against one
reference cannot be used to reproduce that comparison later without
re-running the whole grid. src.eval.runner.write_run now enforces this:
it raises if a metrics subset is missing any of those first three keys.

Usage:
    python scripts/run_xgb_dev.py [--array {array07,array11,array12}]
        [--horizon H] [--regime {lagged,oracle}] [--seed SEED]

Defaults: --array array11 --horizon 3 --regime lagged --seed 0
"""

import argparse
import sys
import time
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
from src.eval.metrics import mae, mbe, nrmse, rmse, skill_score
from src.eval.runner import make_run_id, set_all_seeds, write_run
from src.models.base import check_no_lookahead
from src.models.climatology import Climatology, ConvexCombination
from src.models.persistence import SmartPersistence
from src.models.xgb import XGBForecaster

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

# array key -> (parquet filename, nameplate kW, gamma_pdc). Same three
# arrays and same values as scripts/validate_persistence.py.
ARRAYS = {
    "array07": ("array07_CdTe_hourly.parquet", 7.0, GAMMA_PDC_CDTE),
    "array11": ("array11_polySi_hourly.parquet", 5.0, GAMMA_PDC_SILICON),
    "array12": ("array12_monoSi_hourly.parquet", 5.1, GAMMA_PDC_SILICON),
}

N_TOP_FEATURES = 15


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array", choices=sorted(ARRAYS), default="array11")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--regime", choices=("lagged", "oracle"), default="lagged")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_and_prepare(array_key):
    """Load one array's processed parquet and add every column build_features
    and SmartPersistence require EXCEPT clear-sky power (p_cs, k_p), which
    depends on train-only fitted parameters and is added per-split below.
    """
    filename, nameplate_kw, gamma_pdc = ARRAYS[array_key]
    df = pd.read_parquet(PROCESSED_DIR / filename)
    df = add_solar_position(df)
    df = add_clearsky(df)
    df = add_daylight_mask(df)
    df = add_clearsky_index_ghi(df)
    return df, nameplate_kw, gamma_pdc


def add_clearsky_power_per_split(train, val, test, nameplate_kw, gamma_pdc):
    """Fit the temperature climatology and gain on TRAIN ONLY, then apply
    them to produce p_cs/k_p on train and val. test is left untouched -
    the test split is not read at all in this script.
    """
    temp_clim = fit_temperature_climatology(train)

    p_cs_raw_train = model_clearsky_power(train.index, nameplate_kw, gamma_pdc, temp_clim)
    gain, n_gain_hours, gain_iqr = fit_gain(train, p_cs_raw_train)
    train = add_clearsky_power(train, p_cs_raw_train, gain, nameplate_kw)

    p_cs_raw_val = model_clearsky_power(val.index, nameplate_kw, gamma_pdc, temp_clim)
    val = add_clearsky_power(val, p_cs_raw_val, gain, nameplate_kw)

    return train, val, {"gain": gain, "gain_n_hours": n_gain_hours, "gain_iqr": gain_iqr}


def print_metrics_row(label, m):
    print(
        f"  {label:10s}  MAE={m['mae']:.4f} kW  RMSE={m['rmse']:.4f} kW  "
        f"nRMSE={m['nrmse']:.2f}%  MBE={m['mbe']:+.4f} kW  n={m['n_samples']:>5d}"
    )
    print(
        f"  {'':10s}  skill_vs_persistence={m['skill_vs_persistence']:+.4f}  "
        f"skill_vs_convex={m['skill_vs_convex']:+.4f}  convex_weight={m['convex_weight']:.2f}  "
        f"rmse_persistence={m['rmse_persistence']:.4f}  rmse_convex={m['rmse_convex']:.4f}  "
        f"rmse_climatology={m['rmse_climatology']:.4f}"
    )


def main():
    args = parse_args()
    print(f"array={args.array}  horizon={args.horizon}  regime={args.regime}  seed={args.seed}")

    set_all_seeds(args.seed)

    df, nameplate_kw, gamma_pdc = load_and_prepare(args.array)
    train, val, test = split_chronological(df)
    train, val, gain_info = add_clearsky_power_per_split(train, val, test, nameplate_kw, gamma_pdc)

    print(
        f"gain (fit on train years only): {gain_info['gain']:.4f}  "
        f"(hours used: {gain_info['gain_n_hours']}, IQR: {gain_info['gain_iqr']:.4f})"
    )
    print(f"n_train={len(train)}  n_val={len(val)}  n_test={len(test)} (test not touched)")

    # --- fit XGBoost, val used only for early stopping ---
    model = XGBForecaster(seed=args.seed, regime=args.regime)

    fit_start = time.perf_counter()
    model.fit(train, args.horizon, df_val=val)
    fit_seconds = time.perf_counter() - fit_start
    print(f"fit done in {fit_seconds:.2f}s, best_iteration={model.best_iteration}")

    predict_start = time.perf_counter()
    preds_xgb = model.predict(val, args.horizon)
    predict_seconds = time.perf_counter() - predict_start
    check_no_lookahead(val, preds_xgb, args.horizon)

    # --- reference forecasters, same val split, same horizon ---
    # SmartPersistence: unmodified, see persistence.py.
    sp_model = SmartPersistence()
    sp_model.fit(train, args.horizon)  # no-op, see persistence.py
    preds_sp = sp_model.predict(val, args.horizon)
    check_no_lookahead(val, preds_sp, args.horizon)

    # Climatology: trained-mean k_p per (month, hour), see
    # src/models/climatology.py.
    clim_model = Climatology()
    clim_model.fit(train, args.horizon)
    preds_clim = clim_model.predict(val, args.horizon)

    # ConvexCombination: w*persistence + (1-w)*climatology, w chosen by
    # grid search to minimise RMSE on df_val - CRITICAL, w is fit on
    # VALIDATION only, never on test; see that class's own docstring/
    # comment for why. This is the reference recommended by Yang et al.
    # (2020, Solar Energy 210:20-37) and the one
    # scripts/compare_references.py showed can differ hugely from plain
    # persistence at longer horizons.
    convex_model = ConvexCombination()
    convex_model.fit(train, args.horizon, val)
    preds_convex = convex_model.predict(val, args.horizon)
    check_no_lookahead(val, preds_convex, args.horizon)

    # Restrict every model to the intersection of timestamps where ALL
    # FOUR produced a prediction, so every metric below - the model's own
    # mae/rmse/nrmse/mbe AND both skill scores AND the three reference
    # RMSEs - is computed on the exact same rows. See module docstring.
    common_idx = (
        preds_xgb.index.intersection(preds_sp.index)
        .intersection(preds_clim.index)
        .intersection(preds_convex.index)
    )
    print(
        f"predictions: xgboost={len(preds_xgb)}  smart_persistence={len(preds_sp)}  "
        f"climatology={len(preds_clim)}  convex_reference={len(preds_convex)}  "
        f"intersection={len(common_idx)}"
    )

    y_true = val.loc[common_idx, "Active_Power"]
    y_xgb = preds_xgb.loc[common_idx]
    y_pers = preds_sp.loc[common_idx]
    y_clim = preds_clim.loc[common_idx]
    y_convex = preds_convex.loc[common_idx]
    is_daylight = val.loc[common_idx, "is_daylight"]

    metrics = {}
    for label, mask in [
        ("daylight", is_daylight),
        ("all_hours", pd.Series(True, index=common_idx)),
    ]:
        yt, yx, yp, yc, yv = y_true[mask], y_xgb[mask], y_pers[mask], y_clim[mask], y_convex[mask]
        metrics[label] = {
            "mae": mae(yt, yx),
            "rmse": rmse(yt, yx),
            "nrmse": nrmse(yt, yx, nameplate_kw),
            "mbe": mbe(yt, yx),
            "n_samples": int(mask.sum()),
            "skill_vs_persistence": skill_score(yt, yx, yp),
            "skill_vs_convex": skill_score(yt, yx, yv),
            "convex_weight": convex_model.w,
            "rmse_persistence": rmse(yt, yp),
            "rmse_convex": rmse(yt, yv),
            "rmse_climatology": rmse(yt, yc),
        }

    print(f"\n--- horizon = {args.horizon}h, regime = {args.regime} (validation split, 2014) ---")
    print_metrics_row("daylight", metrics["daylight"])
    print_metrics_row("all_hours", metrics["all_hours"])

    # --- feature importance ---
    importances = model.feature_importance()
    top_features = list(importances.items())[:N_TOP_FEATURES]
    print(f"\ntop {N_TOP_FEATURES} features by gain:")
    for rank, (feat, gain) in enumerate(top_features, start=1):
        print(f"  {rank:2d}. {feat:30s} {gain:12.4f}")

    # --- run record ---
    config = {
        "model": XGBForecaster.name,
        "array": args.array,
        "horizon": args.horizon,
        "regime": args.regime,
        "seed": args.seed,
        # This script never touches 2015 (see module docstring) - every
        # run it writes is a validation-split, development run.
        "eval_split": "val",
        "hyperparams": {
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "subsample": model.subsample,
            "colsample_bytree": model.colsample_bytree,
            "early_stopping_rounds": model.early_stopping_rounds,
        },
        "nameplate_kw": nameplate_kw,
        "best_iteration": model.best_iteration,
        **gain_info,
    }
    timings = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
    }
    extra = {
        "n_preds_xgboost": len(preds_xgb),
        "n_preds_smart_persistence": len(preds_sp),
        "n_preds_climatology": len(preds_clim),
        "n_preds_convex_reference": len(preds_convex),
        "n_preds_intersection": len(common_idx),
        "top_features_gain": dict(top_features),
    }

    out_path = write_run(config, metrics, timings, extra=extra, results_dir=str(RESULTS_DIR))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
