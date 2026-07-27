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
    python scripts/run_xgb_dev.py [--array {array11,array12,array17}]
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

from src.data.pipeline import ARRAYS, add_clearsky_power_per_split, load_and_prepare
from src.data.splits import split_chronological
from src.eval.exclusions import exclusion_mask
from src.eval.metrics import mae, mbe, nrmse, rmse, skill_score
from src.eval.runner import make_run_id, set_all_seeds, write_run
from src.models.base import check_no_lookahead
from src.models.climatology import Climatology, ConvexCombination
from src.models.persistence import SmartPersistence
from src.models.xgb import XGBForecaster

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

N_TOP_FEATURES = 15


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array", choices=sorted(ARRAYS), default="array11")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--regime", choices=("lagged", "oracle"), default="lagged")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


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


def run_experiment(array, horizon, regime, seed, results_dir=RESULTS_DIR, verbose=True):
    """Run one XGBoost dev experiment and write results/<run_id>.json.

    Pulled out of main() so scripts/run_seed_sweep.py can call this
    directly for many (array, horizon, seed) combinations without
    duplicating the pipeline. verbose=False suppresses the per-run prints
    below so a sweep of many runs doesn't flood the console; the caller
    is expected to print its own one-line progress instead.

    Returns (out_path, metrics).
    """

    def vprint(*a, **kw):
        if verbose:
            print(*a, **kw)

    vprint(f"array={array}  horizon={horizon}  regime={regime}  seed={seed}")

    set_all_seeds(seed)

    df, nameplate_kw, gamma_pdc = load_and_prepare(array, PROCESSED_DIR)
    train, val, test = split_chronological(df)
    train, val, gain_info = add_clearsky_power_per_split(train, val, test, nameplate_kw, gamma_pdc)

    vprint(
        f"gain (fit on train years only): {gain_info['gain']:.4f}  "
        f"(hours used: {gain_info['gain_n_hours']}, IQR: {gain_info['gain_iqr']:.4f})"
    )
    vprint(f"n_train={len(train)}  n_val={len(val)}  n_test={len(test)} (test not touched)")

    # --- fit XGBoost, val used only for early stopping ---
    model = XGBForecaster(seed=seed, regime=regime)

    fit_start = time.perf_counter()
    model.fit(train, horizon, df_val=val)
    fit_seconds = time.perf_counter() - fit_start
    vprint(f"fit done in {fit_seconds:.2f}s, best_iteration={model.best_iteration}")

    predict_start = time.perf_counter()
    preds_xgb = model.predict(val, horizon)
    predict_seconds = time.perf_counter() - predict_start
    check_no_lookahead(val, preds_xgb, horizon)

    # --- reference forecasters, same val split, same horizon ---
    # SmartPersistence: unmodified, see persistence.py.
    sp_model = SmartPersistence()
    sp_model.fit(train, horizon)  # no-op, see persistence.py
    preds_sp = sp_model.predict(val, horizon)
    check_no_lookahead(val, preds_sp, horizon)

    # Climatology: trained-mean k_p per (month, hour), see
    # src/models/climatology.py.
    clim_model = Climatology()
    clim_model.fit(train, horizon)
    preds_clim = clim_model.predict(val, horizon)

    # ConvexCombination: w*persistence + (1-w)*climatology, w chosen by
    # grid search to minimise RMSE on df_val - CRITICAL, w is fit on
    # VALIDATION only, never on test; see that class's own docstring/
    # comment for why. This is the reference recommended by Yang et al.
    # (2020, Solar Energy 210:20-37) and the one
    # scripts/compare_references.py showed can differ hugely from plain
    # persistence at longer horizons.
    convex_model = ConvexCombination()
    convex_model.fit(train, horizon, val)
    preds_convex = convex_model.predict(val, horizon)
    check_no_lookahead(val, preds_convex, horizon)

    # Restrict every model to the intersection of timestamps where ALL
    # FOUR produced a prediction, so every metric below - the model's own
    # mae/rmse/nrmse/mbe AND both skill scores AND the three reference
    # RMSEs - is computed on the exact same rows. See module docstring.
    common_idx = (
        preds_xgb.index.intersection(preds_sp.index)
        .intersection(preds_clim.index)
        .intersection(preds_convex.index)
    )
    vprint(
        f"predictions: xgboost={len(preds_xgb)}  smart_persistence={len(preds_sp)}  "
        f"climatology={len(preds_clim)}  convex_reference={len(preds_convex)}  "
        f"intersection={len(common_idx)}"
    )

    # Drop documented-outage timestamps (src.eval.exclusions) before any
    # model's metrics are computed, so every model is evaluated on the
    # identical set of rows and none benefits/suffers from hours known to
    # be equipment-dead rather than a forecasting failure.
    outage_mask = exclusion_mask(array, common_idx)
    n_excluded_outage = int(outage_mask.sum())
    eval_idx = common_idx[~outage_mask]
    if n_excluded_outage:
        vprint(f"excluded {n_excluded_outage} documented-outage hours from evaluation")

    y_true = val.loc[eval_idx, "Active_Power"]
    y_xgb = preds_xgb.loc[eval_idx]
    y_pers = preds_sp.loc[eval_idx]
    y_clim = preds_clim.loc[eval_idx]
    y_convex = preds_convex.loc[eval_idx]
    is_daylight = val.loc[eval_idx, "is_daylight"]

    metrics = {}
    for label, mask in [
        ("daylight", is_daylight),
        ("all_hours", pd.Series(True, index=eval_idx)),
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

    vprint(f"\n--- horizon = {horizon}h, regime = {regime} (validation split, 2014) ---")
    if verbose:
        print_metrics_row("daylight", metrics["daylight"])
        print_metrics_row("all_hours", metrics["all_hours"])

    # --- feature importance ---
    importances = model.feature_importance()
    top_features = list(importances.items())[:N_TOP_FEATURES]
    if verbose:
        print(f"\ntop {N_TOP_FEATURES} features by gain:")
        for rank, (feat, gain) in enumerate(top_features, start=1):
            print(f"  {rank:2d}. {feat:30s} {gain:12.4f}")

    # --- run record ---
    config = {
        "model": XGBForecaster.name,
        "array": array,
        "horizon": horizon,
        "regime": regime,
        "seed": seed,
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
        "n_excluded_outage": n_excluded_outage,
    }
    extra = {
        "n_preds_xgboost": len(preds_xgb),
        "n_preds_smart_persistence": len(preds_sp),
        "n_preds_climatology": len(preds_clim),
        "n_preds_convex_reference": len(preds_convex),
        "n_preds_intersection": len(common_idx),
        "top_features_gain": dict(top_features),
    }

    out_path = write_run(config, metrics, timings, extra=extra, results_dir=str(results_dir))
    vprint(f"\nwrote {out_path}")

    return out_path, metrics


def main():
    args = parse_args()
    run_experiment(args.array, args.horizon, args.regime, args.seed)


if __name__ == "__main__":
    main()
