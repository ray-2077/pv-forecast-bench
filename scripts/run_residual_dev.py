"""Development driver for ResidualCorrected (src/models/residual.py).
Mirrors scripts/run_lstm_dev.py and scripts/run_cnn_lstm_dev.py - same
array/horizon/regime/seed arguments, same VALIDATION-only evaluation, same
reference set (persistence, climatology, convex), same intersection logic,
same outage exclusion, same run JSON schema. Two additions: --base, which
recurrent forecaster the residual stage sits on ('lstm' or 'cnn_lstm' -
LSTMForecaster / CNNLSTMForecaster, constructed here and wrapped in
ResidualCorrected); and --residual-fit-split, see below.

Evaluated on the VALIDATION split (2014) only, NOT the test split (2015) -
per CLAUDE.md's research-integrity rules, the test set is touched once, at
the end of the whole project. This script is model development, so it must
not look at 2015 results at all.

--residual-fit-split {oof,val}, default 'oof' - CLAUDE.md rule 6:
'oof' fits the residual stage on out-of-fold TRAINING residuals (expanding
window over TRAIN_YEARS; VALIDATION used only for early stopping) - this
is the ONLY setting that should produce a reported result, and every other
dev script's caveat-free validation-split metrics apply here too. 'val'
reproduces the ORIGINAL, LEAKED scheme (fit directly on validation
residuals, then evaluated on the same validation split) - it exists only
so its inflation can be measured on purpose; ResidualCorrected raises a
UserWarning when constructed this way and this script records
leaked_by_design=True in config. See src/models/residual.py's module
docstring and CLAUDE.md rule 6's CHANGE LOG for the array11 h6 seed0
evidence (+0.2104 plain LSTM -> +0.5447 leaked) that drove this split.

Pipeline: load one array's processed parquet -> solar position / clear-sky
irradiance / daylight mask / clear-sky index of GHI (src.data.clearsky) ->
chronological split (src.data.splits) -> temperature climatology and gain
fit on TRAIN only, then clear-sky POWER added to train and val
(src.data.clearsky_power) -> ResidualCorrected.fit(train, horizon,
df_val=val), which internally fits the base model on all of train (val for
early stopping) and then the residual XGBoost per --residual-fit-split ->
predict on val -> metrics against SmartPersistence AND the
ConvexCombination reference (src.models.climatology - weight fitted on
VALIDATION, per that module's own CRITICAL comment) on the same rows,
daylight-only and all-hours -> results/<run_id>.json.

Every run also records skill_vs_persistence, skill_vs_convex,
convex_weight, rmse_persistence, rmse_convex, rmse_climatology in each
metrics subset - see src.eval.runner.write_run, which enforces this.

Usage:
    python scripts/run_residual_dev.py [--base {lstm,cnn_lstm}]
        [--array {array11,array12,array17}] [--horizon H]
        [--regime {lagged,oracle}] [--seed SEED]
        [--residual-fit-split {oof,val}]

Defaults: --base lstm --array array11 --horizon 3 --regime lagged --seed 0
          --residual-fit-split oof
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.pipeline import ARRAYS, add_clearsky_power_per_split, load_and_prepare
from src.data.splits import TRAIN_YEARS, split_chronological
from src.eval.exclusions import exclusion_mask
from src.eval.metrics import mae, mbe, nrmse, rmse, skill_score
from src.eval.runner import make_run_id, set_all_seeds, write_run
from src.models.base import check_no_lookahead
from src.models.climatology import Climatology, ConvexCombination
from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.lstm import LSTMForecaster
from src.models.persistence import SmartPersistence
from src.models.residual import ResidualCorrected

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

N_TOP_FEATURES = 15

BASE_MODEL_CLASSES = {
    "lstm": LSTMForecaster,
    "cnn_lstm": CNNLSTMForecaster,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", choices=sorted(BASE_MODEL_CLASSES), default="lstm")
    parser.add_argument("--array", choices=sorted(ARRAYS), default="array11")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--regime", choices=("lagged", "oracle"), default="lagged")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--residual-fit-split", choices=("oof", "val"), default="oof")
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


def run_experiment(
    array,
    horizon,
    regime,
    seed,
    base="lstm",
    residual_fit_split="oof",
    train_years=None,
    results_dir=RESULTS_DIR,
    verbose=True,
):
    """Run one residual-corrected dev experiment and write
    results/<run_id>.json.

    Argument order matches the other dev scripts' run_experiment(array,
    horizon, regime, seed, verbose=...) so scripts/run_seed_sweep.py can
    call this the same way, with `base` bound via functools.partial for
    each of the two MODEL_RUNNERS entries (lstm_residual, cnn_lstm_residual)
    it registers - both use the residual_fit_split='oof' default, per
    CLAUDE.md rule 6.

    train_years=None (default) uses src.data.splits.TRAIN_YEARS, i.e.
    identical behaviour to before this parameter existed. Passing an
    explicit tuple overrides the training window (VAL_YEARS/TEST_YEARS are
    never touched) - this also changes the fold count inside
    ResidualCorrected._oof_residuals, which now derives fold years from the
    actual training data rather than the hardcoded TRAIN_YEARS constant
    (see src/models/residual.py). Used by scripts/rerun_residual_5yr.py for
    the 4-fold ablation. The effective window is recorded in
    config["train_years"].

    Returns (out_path, metrics).
    """

    def vprint(*a, **kw):
        if verbose:
            print(*a, **kw)

    effective_train_years = tuple(train_years) if train_years is not None else TRAIN_YEARS

    vprint(
        f"base={base}  residual_fit_split={residual_fit_split}  array={array}  "
        f"horizon={horizon}  regime={regime}  seed={seed}  "
        f"train_years={effective_train_years}"
    )

    set_all_seeds(seed)

    df, nameplate_kw, gamma_pdc = load_and_prepare(array, PROCESSED_DIR)
    train, val, test = split_chronological(df, train_years=effective_train_years)
    train, val, gain_info = add_clearsky_power_per_split(train, val, test, nameplate_kw, gamma_pdc)

    vprint(
        f"gain (fit on train years only): {gain_info['gain']:.4f}  "
        f"(hours used: {gain_info['gain_n_hours']}, IQR: {gain_info['gain_iqr']:.4f})"
    )
    vprint(f"n_train={len(train)}  n_val={len(val)}  n_test={len(test)} (test not touched)")

    # --- fit base model on ALL of train (val for early stopping), then the
    # residual XGBoost per residual_fit_split - both inside
    # ResidualCorrected.fit; see src/models/residual.py's docstring and
    # CLAUDE.md rule 6 for the out-of-fold scheme ('oof', the default) and
    # why 'val' is leaked and must not be used for a reported result. ---
    base_model = BASE_MODEL_CLASSES[base](seed=seed, regime=regime)
    residual_model = ResidualCorrected(base_model, seed=seed, residual_fit_split=residual_fit_split)
    vprint(f"base device={base_model.device}")

    fit_start = time.perf_counter()
    residual_model.fit(train, horizon, val)
    fit_seconds = time.perf_counter() - fit_start
    vprint(
        f"fit done in {fit_seconds:.2f}s, base_best_epoch={base_model.best_epoch}, "
        f"base_epochs_run={base_model.epochs_run}, "
        f"base_full_determinism_achieved={base_model.full_determinism_achieved}"
    )
    if residual_model.residual_fit_split == "oof":
        vprint(
            f"residual stage (out-of-fold, TRAIN_YEARS): "
            f"n_oof_residuals={residual_model.n_oof_residuals}  "
            f"years_used={residual_model.oof_years}"
        )
    else:
        vprint(
            f"residual stage (LEAKED, val-fit - see CLAUDE.md rule 6): "
            f"n_val_residuals={residual_model.n_val_residuals}  "
            f"frac_val_usable={residual_model.frac_val_usable:.4f}"
        )

    predict_start = time.perf_counter()
    preds_residual = residual_model.predict(val, horizon)
    predict_seconds = time.perf_counter() - predict_start
    check_no_lookahead(val, preds_residual, horizon)

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
        preds_residual.index.intersection(preds_sp.index)
        .intersection(preds_clim.index)
        .intersection(preds_convex.index)
    )
    vprint(
        f"predictions: {residual_model.name}={len(preds_residual)}  "
        f"smart_persistence={len(preds_sp)}  climatology={len(preds_clim)}  "
        f"convex_reference={len(preds_convex)}  intersection={len(common_idx)}"
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
    y_residual = preds_residual.loc[eval_idx]
    y_pers = preds_sp.loc[eval_idx]
    y_clim = preds_clim.loc[eval_idx]
    y_convex = preds_convex.loc[eval_idx]
    is_daylight = val.loc[eval_idx, "is_daylight"]

    metrics = {}
    for label, mask in [
        ("daylight", is_daylight),
        ("all_hours", pd.Series(True, index=eval_idx)),
    ]:
        yt, yr, yp, yc, yv = y_true[mask], y_residual[mask], y_pers[mask], y_clim[mask], y_convex[mask]
        metrics[label] = {
            "mae": mae(yt, yr),
            "rmse": rmse(yt, yr),
            "nrmse": nrmse(yt, yr, nameplate_kw),
            "mbe": mbe(yt, yr),
            "n_samples": int(mask.sum()),
            "skill_vs_persistence": skill_score(yt, yr, yp),
            "skill_vs_convex": skill_score(yt, yr, yv),
            "convex_weight": convex_model.w,
            "rmse_persistence": rmse(yt, yp),
            "rmse_convex": rmse(yt, yv),
            "rmse_climatology": rmse(yt, yc),
        }

    vprint(f"\n--- horizon = {horizon}h, regime = {regime}, base = {base} (validation split, 2014) ---")
    if verbose:
        print_metrics_row("daylight", metrics["daylight"])
        print_metrics_row("all_hours", metrics["all_hours"])

    # --- residual feature importance ---
    importances = residual_model.residual_importance()
    top_features = list(importances.items())[:N_TOP_FEATURES]
    if verbose:
        print(f"\ntop {N_TOP_FEATURES} residual features by gain:")
        for rank, (feat, gain) in enumerate(top_features, start=1):
            print(f"  {rank:2d}. {feat:30s} {gain:12.4f}")

    # --- run record ---
    hyperparams = {
        "base_hidden_size": base_model.hidden_size,
        "base_num_layers": base_model.num_layers,
        "base_dropout": base_model.dropout,
        "base_seq_len": base_model.seq_len,
        "base_batch_size": base_model.batch_size,
        "base_max_epochs": base_model.max_epochs,
        "base_patience": base_model.patience,
        "base_learning_rate": base_model.learning_rate,
        "residual_n_estimators": residual_model.n_estimators,
        "residual_max_depth": residual_model.max_depth,
        "residual_learning_rate": residual_model.learning_rate,
        "residual_subsample": residual_model.subsample,
        "residual_colsample_bytree": residual_model.colsample_bytree,
    }
    if base == "cnn_lstm":
        hyperparams["base_n_filters"] = base_model.n_filters
        hyperparams["base_kernel_size"] = base_model.kernel_size

    config = {
        "model": residual_model.name,
        "base_model": base,
        "residual_fit_split": residual_model.residual_fit_split,
        # True only for residual_fit_split='val' - the deliberately-leaked
        # scheme (CLAUDE.md rule 6 CHANGE LOG). Any run with this True must
        # never be treated as a reported result - see
        # src/models/residual.py's module docstring.
        "leaked_by_design": residual_model.leaked_by_design,
        "array": array,
        "horizon": horizon,
        "regime": regime,
        "seed": seed,
        # This script never touches 2015 (see module docstring) - every
        # run it writes is a validation-split, development run.
        "eval_split": "val",
        "train_years": list(effective_train_years),
        "hyperparams": hyperparams,
        "nameplate_kw": nameplate_kw,
        "base_best_epoch": base_model.best_epoch,
        "base_epochs_run": base_model.epochs_run,
        "base_full_determinism_achieved": base_model.full_determinism_achieved,
        "device": str(base_model.device),
        **gain_info,
    }
    if residual_model.residual_fit_split == "oof":
        config["n_oof_residuals"] = residual_model.n_oof_residuals
        config["oof_years"] = residual_model.oof_years
    else:
        config["n_val_residuals"] = residual_model.n_val_residuals
        config["frac_val_usable"] = residual_model.frac_val_usable
    timings = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_excluded_outage": n_excluded_outage,
    }
    extra = {
        f"n_preds_{residual_model.name}": len(preds_residual),
        "n_preds_smart_persistence": len(preds_sp),
        "n_preds_climatology": len(preds_clim),
        "n_preds_convex_reference": len(preds_convex),
        "n_preds_intersection": len(common_idx),
        "top_residual_features_gain": dict(top_features),
        "base_history": base_model.history,
    }

    out_path = write_run(config, metrics, timings, extra=extra, results_dir=str(results_dir))
    vprint(f"\nwrote {out_path}")

    return out_path, metrics


def main():
    args = parse_args()
    run_experiment(
        args.array,
        args.horizon,
        args.regime,
        args.seed,
        base=args.base,
        residual_fit_split=args.residual_fit_split,
    )


if __name__ == "__main__":
    main()
