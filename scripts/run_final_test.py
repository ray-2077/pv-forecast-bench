"""THE FINAL TEST-SET EVALUATION. Test split (2015) is touched exactly
once, per CLAUDE.md's research-integrity rule, after every modelling
and evaluation choice is frozen. THIS SCRIPT IS WRITE-ONCE: once it has
been run with --confirm for a real result, do not edit it and re-run to
get a different number. If a hyperparameter or protocol choice looks
suboptimal against the test numbers, that is a finding for the paper,
not a reason to change this script, the models it calls, or the
protocol, and re-run. Every run this script writes carries
eval_split="test" and the git commit hash it ran at, so a later reader
can see exactly what produced it.

DOES NOT TOUCH ANY SCRIPT THAT PRODUCED A VALIDATION RESULT. This
script does not import, call, or modify run_xgb_dev.py, run_lstm_dev.py,
run_cnn_lstm_dev.py, run_residual_dev.py, or run_seed_sweep.py - all
five hardcode evaluation on the VALIDATION split with no parameter to
redirect that, and editing any of them to add one would be exactly the
kind of touch the brief for this script forbids. Instead this script
re-implements the same fit -> evaluate -> write pipeline directly and
independently, importing only the underlying library code every dev
script already imports too (src.data.*, src.models.*, src.eval.*) -
same model classes, same UNCHANGED default hyperparameters (nothing is
retuned here - see probe_hyperparams, which reads them live off the
classes rather than retyping numbers by hand), same reference
forecasters, same leakage/exclusion code. The one thing that differs
from the val pipeline: the final predict() + metrics call targets TEST
(2015) instead of VAL (2014). VAL is still used for early stopping and
for fitting the convex weight w, exactly as in every validation run.

CLEAR-SKY POWER ON TEST: src.data.pipeline.add_clearsky_power_per_split
deliberately only touches train/val - its own docstring says "test is
left untouched: callers must not read the test split at all in a dev
script". That refusal is correct for every dev script; it is also
exactly the one thing this script is allowed to do once. Rather than
touch that function, add_clearsky_power_all_splits() below calls the
same underlying src.data.clearsky_power primitives it calls
(fit_temperature_climatology, model_clearsky_power, fit_gain,
add_clearsky_power) one extra time, applying the SAME train-only-fitted
temperature climatology and gain to test - never refitting anything on
val or test.

OUTPUT PATH: results/test/<run_id>.json, NOT results/<run_id>.json.
make_run_id() (src/eval/runner.py) does not encode eval_split in the
run id string - it is model_array_hH_regime_seedN, identical whether
eval_split is "val" or "test" - so writing test results into the same
results/ directory under the same naming scheme would either collide
with (and, without overwrite=True, safely refuse to touch) the existing
validation JSON of the same name, or silently invite a future mistake.
A dedicated results/test/ subdirectory with the same run_id naming
makes "this is the test-split counterpart of results/<run_id>.json"
unambiguous from the path alone, requires no change to
src/eval/runner.py, and reuses write_run() completely unmodified
(results_dir is already one of its parameters). This is a design
decision made without asking first because the collision it avoids
would otherwise be silently destructive; flagged here and in the
frozen-config block so it can be overridden before the first real run
if a different convention is wanted.

TWO-PHASE, CONFIRM-GATED: running this script with no arguments ONLY
prints the frozen-configuration block below and exits - it fits
nothing, evaluates nothing, writes nothing. Pass --confirm to actually
execute the 450-run grid. This is deliberate: the brief for this
script requires printing the frozen configuration and stopping for
confirmation before anything runs, and a CLI flag makes "confirm, then
run" a distinct, auditable second invocation rather than a prompt
embedded in one non-interactive process.

DIRTY-TREE GUARD: if git_dirty is True, the script refuses to proceed
under --confirm too, unconditionally - a commit hash recorded while the
tree is dirty does not describe the code that actually ran (see
src.eval.runner.capture_environment's own docstring), which is
disqualifying for a write-once final result.

SKIP-IF-EXISTS: like scripts/run_seed_sweep.py, a run whose
results/test/<run_id>.json already exists is skipped, not recomputed or
overwritten - write_run's own FileExistsError guard (overwrite=False,
the default) is what actually enforces this; this script relies on it
rather than re-implementing the check, same as run_seed_sweep.py does
for validation runs.

Grid: 5 models (xgboost, lstm, cnn_lstm, lstm_residual,
cnn_lstm_residual) x 2 regimes (lagged, oracle) x 3 arrays x 3 horizons
x 5 seeds = 450 runs - the same shape as the combined 225-lagged +
225-oracle validation sweep, so the wall-clock estimate below is drawn
directly from the 450 already-committed validation run JSONs' own
fit_seconds/predict_seconds, not guessed.

Usage:
    python scripts/run_final_test.py             # prints frozen config, exits
    python scripts/run_final_test.py --confirm    # prints frozen config, then runs
"""

import argparse
import glob
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.clearsky import add_daylight_mask
from src.data.clearsky_power import (
    add_clearsky_power,
    fit_gain,
    fit_temperature_climatology,
    model_clearsky_power,
)
from src.data.pipeline import ARRAYS, load_and_prepare
from src.data.splits import TEST_YEARS, TRAIN_YEARS, VAL_YEARS, split_chronological
from src.eval.exclusions import KNOWN_OUTAGES, exclusion_mask
from src.eval.metrics import mae, mbe, nrmse, rmse, skill_score
from src.eval.runner import make_run_id, set_all_seeds, write_run
from src.models.base import check_no_lookahead
from src.models.climatology import _W_GRID, Climatology, ConvexCombination
from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.lstm import LSTMForecaster
from src.models.persistence import SmartPersistence
from src.models.residual import ResidualCorrected
from src.models.xgb import XGBForecaster

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"
TEST_RESULTS_DIR = RESULTS_DIR / "test"

MODELS = ["xgboost", "lstm", "cnn_lstm", "lstm_residual", "cnn_lstm_residual"]
REGIMES = ["lagged", "oracle"]
HORIZONS = [1, 3, 6]
SEEDS = [0, 1, 2, 3, 4]

N_TOP_FEATURES = 15


# ---------------------------------------------------------------------
# clear-sky power, extended to test - see module docstring
# ---------------------------------------------------------------------

def add_clearsky_power_all_splits(train, val, test, nameplate_kw, gamma_pdc):
    """Same calls as src.data.pipeline.add_clearsky_power_per_split
    (fit_temperature_climatology on TRAIN, model_clearsky_power +
    fit_gain on TRAIN, add_clearsky_power on train/val), extended to
    apply that same train-only-fitted temp_clim and gain to test too -
    the one place in this project allowed to do that. Returns
    (train, val, test, gain_info).
    """
    temp_clim = fit_temperature_climatology(train)

    p_cs_raw_train = model_clearsky_power(train.index, nameplate_kw, gamma_pdc, temp_clim)
    gain, n_gain_hours, gain_iqr = fit_gain(train, p_cs_raw_train)
    train = add_clearsky_power(train, p_cs_raw_train, gain, nameplate_kw)

    p_cs_raw_val = model_clearsky_power(val.index, nameplate_kw, gamma_pdc, temp_clim)
    val = add_clearsky_power(val, p_cs_raw_val, gain, nameplate_kw)

    p_cs_raw_test = model_clearsky_power(test.index, nameplate_kw, gamma_pdc, temp_clim)
    test = add_clearsky_power(test, p_cs_raw_test, gain, nameplate_kw)

    gain_info = {"gain": gain, "gain_n_hours": n_gain_hours, "gain_iqr": gain_iqr}
    return train, val, test, gain_info


# ---------------------------------------------------------------------
# frozen-configuration reporting
# ---------------------------------------------------------------------

def git_state():
    def run(*args):
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    commit = run("rev-parse", "HEAD")
    dirty = bool(run("status", "--porcelain"))
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    return commit, dirty, branch


def probe_hyperparams():
    """Instantiate each model class once with its own defaults (never
    overridden anywhere in this script) and read back the
    hyperparameters that will actually be used - printed, not retyped
    by hand, so this cannot silently drift from the real constructor
    defaults.
    """
    xgb = XGBForecaster(seed=0)
    lstm = LSTMForecaster(seed=0)
    cnn_lstm = CNNLSTMForecaster(seed=0)
    lstm_residual = ResidualCorrected(LSTMForecaster(seed=0), seed=0)
    cnn_lstm_residual = ResidualCorrected(CNNLSTMForecaster(seed=0), seed=0)

    return {
        "xgboost": {
            "n_estimators": xgb.n_estimators,
            "max_depth": xgb.max_depth,
            "learning_rate": xgb.learning_rate,
            "subsample": xgb.subsample,
            "colsample_bytree": xgb.colsample_bytree,
            "early_stopping_rounds": xgb.early_stopping_rounds,
        },
        "lstm": {
            "hidden_size": lstm.hidden_size,
            "num_layers": lstm.num_layers,
            "dropout": lstm.dropout,
            "seq_len": lstm.seq_len,
            "batch_size": lstm.batch_size,
            "max_epochs": lstm.max_epochs,
            "patience": lstm.patience,
            "learning_rate": lstm.learning_rate,
        },
        "cnn_lstm": {
            "n_filters": cnn_lstm.n_filters,
            "kernel_size": cnn_lstm.kernel_size,
            "hidden_size": cnn_lstm.hidden_size,
            "num_layers": cnn_lstm.num_layers,
            "dropout": cnn_lstm.dropout,
            "seq_len": cnn_lstm.seq_len,
            "batch_size": cnn_lstm.batch_size,
            "max_epochs": cnn_lstm.max_epochs,
            "patience": cnn_lstm.patience,
            "learning_rate": cnn_lstm.learning_rate,
        },
        "lstm_residual (base + residual stage)": {
            "base": "LSTM, same hyperparameters as 'lstm' above",
            "residual_fit_split": lstm_residual.residual_fit_split,
            "residual_n_estimators": lstm_residual.n_estimators,
            "residual_max_depth": lstm_residual.max_depth,
            "residual_learning_rate": lstm_residual.learning_rate,
            "residual_subsample": lstm_residual.subsample,
            "residual_colsample_bytree": lstm_residual.colsample_bytree,
        },
        "cnn_lstm_residual (base + residual stage)": {
            "base": "CNN-LSTM, same hyperparameters as 'cnn_lstm' above",
            "residual_fit_split": cnn_lstm_residual.residual_fit_split,
            "residual_n_estimators": cnn_lstm_residual.n_estimators,
            "residual_max_depth": cnn_lstm_residual.max_depth,
            "residual_learning_rate": cnn_lstm_residual.learning_rate,
            "residual_subsample": cnn_lstm_residual.subsample,
            "residual_colsample_bytree": cnn_lstm_residual.colsample_bytree,
        },
    }


def estimate_wallclock_seconds():
    """Sum of fit_seconds + predict_seconds across all 450 already-
    committed validation run JSONs (both regimes) - the test grid is
    the same shape, so this is a measured estimate, not a guess. Adds a
    fixed per-run data-loading/feature-build overhead (~9.5s/run, per
    PROJECT_CHECKPOINT.md Section 6's own measurement from sweep-log
    wall time minus summed fit_seconds+predict_seconds) since that part
    is not recorded per-run in timings.
    """
    total_fit = 0.0
    total_predict = 0.0
    n = 0
    for regime in REGIMES:
        for fp in glob.glob(str(RESULTS_DIR / f"*_array*_h*_{regime}_seed*.json")):
            if "leaked" in fp:
                continue
            with open(fp) as fh:
                d = json.load(fh)
            total_fit += d["timings"]["fit_seconds"]
            total_predict += d["timings"]["predict_seconds"]
            n += 1
    overhead_per_run = 9.5
    return total_fit + total_predict + n * overhead_per_run, n


def print_frozen_config():
    commit, dirty, branch = git_state()
    hyperparams = probe_hyperparams()
    est_seconds, n_measured = estimate_wallclock_seconds()

    print("=" * 78)
    print("FROZEN CONFIGURATION - scripts/run_final_test.py")
    print("=" * 78)

    print("\n--- splits (src/data/splits.py, chronological, never shuffled) ---")
    print(f"  TRAIN_YEARS = {TRAIN_YEARS}")
    print(f"  VAL_YEARS   = {VAL_YEARS}   (early stopping, convex weight w only)")
    print(f"  TEST_YEARS  = {TEST_YEARS}   (this script - touched once, at the end)")

    print("\n--- models and hyperparameters (read from live class defaults,")
    print("    never overridden here) ---")
    for name, params in hyperparams.items():
        print(f"  {name}:")
        for k, v in params.items():
            print(f"    {k}: {v}")

    print("\n--- reference forecasters ---")
    print(f"  SmartPersistence.FFILL_LIMIT_HOURS = {SmartPersistence.FFILL_LIMIT_HOURS}")
    print("  Climatology: mean k_p per (month, hour) fit on TRAIN only, no hyperparameters")
    print(
        f"  ConvexCombination: w = argmin RMSE(w*persistence + (1-w)*climatology) "
        f"over VAL, grid step {_W_GRID[1] - _W_GRID[0]:.2f}, range "
        f"[{_W_GRID[0]:.2f}, {_W_GRID[-1]:.2f}] - refit per (array, horizon, seed, regime) cell"
    )

    print("\n--- daylight filter (src/data/clearsky.py add_daylight_mask) ---")
    default_elevation = inspect.signature(add_daylight_mask).parameters["min_elevation"].default
    print(f"  solar_elevation > {default_elevation} deg")

    print("\n--- outage exclusions (src/eval/exclusions.py KNOWN_OUTAGES) ---")
    for (array, start, end), reason in KNOWN_OUTAGES.items():
        note = "  <-- INSIDE TEST YEAR 2015" if start.startswith("2015") else ""
        print(f"  {array}: {start} to {end} (inclusive){note}")
        print(f"    reason: {reason}")

    print("\n--- grid ---")
    print(f"  models:   {MODELS}")
    print(f"  regimes:  {REGIMES}")
    print(f"  horizons: {HORIZONS}")
    print(f"  arrays:   {list(ARRAYS)}")
    print(f"  seeds:    {SEEDS}")
    total_runs = len(MODELS) * len(REGIMES) * len(HORIZONS) * len(ARRAYS) * len(SEEDS)
    print(
        f"  total: {len(MODELS)} x {len(REGIMES)} x {len(HORIZONS)} x "
        f"{len(ARRAYS)} x {len(SEEDS)} = {total_runs} runs"
    )

    print("\n--- output path convention (see module docstring) ---")
    print("  results/test/<run_id>.json, NOT results/<run_id>.json")
    print(f"  e.g. {TEST_RESULTS_DIR / 'xgboost_array11_h3_lagged_seed0.json'}")
    print("  (make_run_id does not encode eval_split; a shared results/ directory")
    print("   would collide with the existing validation JSON of the same name)")

    print("\n--- git state ---")
    print(f"  commit: {commit}")
    print(f"  branch: {branch}")
    print(f"  dirty:  {dirty}")
    if dirty:
        print(
            "\n  ABORT CONDITION: working tree is dirty. A commit hash recorded "
            "while dirty does not describe the code that actually ran - this "
            "script will refuse to execute even with --confirm until the tree "
            "is clean (committed or stashed)."
        )

    est_minutes = est_seconds / 60
    est_hours = est_minutes / 60
    print("\n--- estimated wall-clock ---")
    print(
        f"  based on summed fit_seconds+predict_seconds across all "
        f"{n_measured} already-committed validation runs (same grid shape), "
        f"plus ~9.5s/run measured data-loading overhead:"
    )
    print(f"  ~{est_minutes:.0f} min (~{est_hours:.1f} hours) for {total_runs} runs")

    print("\n" + "=" * 78)
    return dirty


# ---------------------------------------------------------------------
# shared evaluation tail - identical in every dev script this mirrors:
# reference forecasters, four-way prediction intersection, outage
# exclusion, metrics block. The only change from the val versions this
# is derived from: `eval_df` is TEST, not VAL.
# ---------------------------------------------------------------------

def _reference_forecasters(train, val, horizon):
    sp_model = SmartPersistence()
    sp_model.fit(train, horizon)
    clim_model = Climatology()
    clim_model.fit(train, horizon)
    convex_model = ConvexCombination()
    convex_model.fit(train, horizon, val)  # w fit on VAL only, never test
    return sp_model, clim_model, convex_model


def _evaluate_on_test(
    model_preds, eval_df, array, horizon, nameplate_kw, sp_model, clim_model, convex_model
):
    preds_sp = sp_model.predict(eval_df, horizon)
    check_no_lookahead(eval_df, preds_sp, horizon)
    preds_clim = clim_model.predict(eval_df, horizon)
    preds_convex = convex_model.predict(eval_df, horizon)
    check_no_lookahead(eval_df, preds_convex, horizon)

    common_idx = (
        model_preds.index.intersection(preds_sp.index)
        .intersection(preds_clim.index)
        .intersection(preds_convex.index)
    )

    outage_mask = exclusion_mask(array, common_idx)
    n_excluded_outage = int(outage_mask.sum())
    eval_idx = common_idx[~outage_mask]

    y_true = eval_df.loc[eval_idx, "Active_Power"]
    y_model = model_preds.loc[eval_idx]
    y_pers = preds_sp.loc[eval_idx]
    y_clim = preds_clim.loc[eval_idx]
    y_convex = preds_convex.loc[eval_idx]
    is_daylight = eval_df.loc[eval_idx, "is_daylight"]

    metrics = {}
    for label, mask in [
        ("daylight", is_daylight),
        ("common_hours", pd.Series(True, index=eval_idx)),
    ]:
        yt, ym, yp, yc, yv = y_true[mask], y_model[mask], y_pers[mask], y_clim[mask], y_convex[mask]
        metrics[label] = {
            "mae": mae(yt, ym),
            "rmse": rmse(yt, ym),
            "nrmse": nrmse(yt, ym, nameplate_kw),
            "mbe": mbe(yt, ym),
            "n_samples": int(mask.sum()),
            "skill_vs_persistence": skill_score(yt, ym, yp),
            "skill_vs_convex": skill_score(yt, ym, yv),
            "convex_weight": convex_model.w,
            "rmse_persistence": rmse(yt, yp),
            "rmse_convex": rmse(yt, yv),
            "rmse_climatology": rmse(yt, yc),
        }

    n_preds = {
        "n_preds_smart_persistence": len(preds_sp),
        "n_preds_climatology": len(preds_clim),
        "n_preds_convex_reference": len(preds_convex),
        "n_preds_intersection": len(common_idx),
    }
    return metrics, n_excluded_outage, n_preds


def _load_and_split(array):
    """load_and_prepare + split_chronological + clear-sky power on all
    three splits (see add_clearsky_power_all_splits). Shared by every
    test_<model> function below.
    """
    df, nameplate_kw, gamma_pdc = load_and_prepare(array, PROCESSED_DIR)
    train, val, test = split_chronological(df)
    train, val, test, gain_info = add_clearsky_power_all_splits(
        train, val, test, nameplate_kw, gamma_pdc
    )
    return train, val, test, nameplate_kw, gain_info


# ---------------------------------------------------------------------
# per-model test runs - each mirrors its dev-script counterpart's fit
# logic (scripts/run_xgb_dev.py, run_lstm_dev.py, run_cnn_lstm_dev.py,
# run_residual_dev.py - none of which this script imports or edits),
# predicting on TEST instead of VAL for the final evaluation step only.
# ---------------------------------------------------------------------

def test_xgboost(array, horizon, regime, seed):
    set_all_seeds(seed)
    train, val, test, nameplate_kw, gain_info = _load_and_split(array)

    model = XGBForecaster(seed=seed, regime=regime)
    fit_start = time.perf_counter()
    model.fit(train, horizon, df_val=val)
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    preds = model.predict(test, horizon)
    predict_seconds = time.perf_counter() - predict_start
    check_no_lookahead(test, preds, horizon)

    sp_model, clim_model, convex_model = _reference_forecasters(train, val, horizon)
    metrics, n_excluded_outage, n_preds = _evaluate_on_test(
        preds, test, array, horizon, nameplate_kw, sp_model, clim_model, convex_model
    )

    top_features = list(model.feature_importance().items())[:N_TOP_FEATURES]

    config = {
        "model": XGBForecaster.name,
        "array": array,
        "horizon": horizon,
        "regime": regime,
        "seed": seed,
        "eval_split": "test",
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
    extra = {"n_preds_xgboost": len(preds), **n_preds, "top_features_gain": dict(top_features)}
    return config, metrics, timings, extra


def test_lstm(array, horizon, regime, seed):
    set_all_seeds(seed)
    train, val, test, nameplate_kw, gain_info = _load_and_split(array)

    model = LSTMForecaster(seed=seed, regime=regime)
    fit_start = time.perf_counter()
    model.fit(train, horizon, df_val=val)
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    preds = model.predict(test, horizon)
    predict_seconds = time.perf_counter() - predict_start
    check_no_lookahead(test, preds, horizon)

    sp_model, clim_model, convex_model = _reference_forecasters(train, val, horizon)
    metrics, n_excluded_outage, n_preds = _evaluate_on_test(
        preds, test, array, horizon, nameplate_kw, sp_model, clim_model, convex_model
    )

    config = {
        "model": LSTMForecaster.name,
        "array": array,
        "horizon": horizon,
        "regime": regime,
        "seed": seed,
        "eval_split": "test",
        "hyperparams": {
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "dropout": model.dropout,
            "seq_len": model.seq_len,
            "batch_size": model.batch_size,
            "max_epochs": model.max_epochs,
            "patience": model.patience,
            "learning_rate": model.learning_rate,
        },
        "nameplate_kw": nameplate_kw,
        "best_epoch": model.best_epoch,
        "epochs_run": model.epochs_run,
        "full_determinism_achieved": model.full_determinism_achieved,
        "device": str(model.device),
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
    extra = {"n_preds_lstm": len(preds), **n_preds, "history": model.history}
    return config, metrics, timings, extra


def test_cnn_lstm(array, horizon, regime, seed):
    set_all_seeds(seed)
    train, val, test, nameplate_kw, gain_info = _load_and_split(array)

    model = CNNLSTMForecaster(seed=seed, regime=regime)
    fit_start = time.perf_counter()
    model.fit(train, horizon, df_val=val)
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    preds = model.predict(test, horizon)
    predict_seconds = time.perf_counter() - predict_start
    check_no_lookahead(test, preds, horizon)

    sp_model, clim_model, convex_model = _reference_forecasters(train, val, horizon)
    metrics, n_excluded_outage, n_preds = _evaluate_on_test(
        preds, test, array, horizon, nameplate_kw, sp_model, clim_model, convex_model
    )

    config = {
        "model": CNNLSTMForecaster.name,
        "array": array,
        "horizon": horizon,
        "regime": regime,
        "seed": seed,
        "eval_split": "test",
        "hyperparams": {
            "n_filters": model.n_filters,
            "kernel_size": model.kernel_size,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "dropout": model.dropout,
            "seq_len": model.seq_len,
            "batch_size": model.batch_size,
            "max_epochs": model.max_epochs,
            "patience": model.patience,
            "learning_rate": model.learning_rate,
        },
        "nameplate_kw": nameplate_kw,
        "best_epoch": model.best_epoch,
        "epochs_run": model.epochs_run,
        "full_determinism_achieved": model.full_determinism_achieved,
        "device": str(model.device),
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
    extra = {"n_preds_cnn_lstm": len(preds), **n_preds, "history": model.history}
    return config, metrics, timings, extra


BASE_MODEL_CLASSES = {"lstm": LSTMForecaster, "cnn_lstm": CNNLSTMForecaster}


def test_residual(array, horizon, regime, seed, base):
    set_all_seeds(seed)
    train, val, test, nameplate_kw, gain_info = _load_and_split(array)

    base_model = BASE_MODEL_CLASSES[base](seed=seed, regime=regime)
    residual_model = ResidualCorrected(base_model, seed=seed, residual_fit_split="oof")

    fit_start = time.perf_counter()
    residual_model.fit(train, horizon, val)
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    preds = residual_model.predict(test, horizon)
    predict_seconds = time.perf_counter() - predict_start
    check_no_lookahead(test, preds, horizon)

    sp_model, clim_model, convex_model = _reference_forecasters(train, val, horizon)
    metrics, n_excluded_outage, n_preds = _evaluate_on_test(
        preds, test, array, horizon, nameplate_kw, sp_model, clim_model, convex_model
    )

    top_features = list(residual_model.residual_importance().items())[:N_TOP_FEATURES]

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
        "leaked_by_design": residual_model.leaked_by_design,
        "array": array,
        "horizon": horizon,
        "regime": regime,
        "seed": seed,
        "eval_split": "test",
        "hyperparams": hyperparams,
        "nameplate_kw": nameplate_kw,
        "base_best_epoch": base_model.best_epoch,
        "base_epochs_run": base_model.epochs_run,
        "base_full_determinism_achieved": base_model.full_determinism_achieved,
        "device": str(base_model.device),
        "n_oof_residuals": residual_model.n_oof_residuals,
        "oof_years": residual_model.oof_years,
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
        f"n_preds_{residual_model.name}": len(preds),
        **n_preds,
        "top_residual_features_gain": dict(top_features),
        "base_history": base_model.history,
    }
    return config, metrics, timings, extra


MODEL_RUNNERS = {
    "xgboost": test_xgboost,
    "lstm": test_lstm,
    "cnn_lstm": test_cnn_lstm,
    "lstm_residual": lambda array, horizon, regime, seed: test_residual(
        array, horizon, regime, seed, base="lstm"
    ),
    "cnn_lstm_residual": lambda array, horizon, regime, seed: test_residual(
        array, horizon, regime, seed, base="cnn_lstm"
    ),
}


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="actually execute the 450-run grid (default: print frozen config and exit)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dirty = print_frozen_config()

    if dirty:
        print("\nABORTING: working tree is dirty. Commit or stash before running.")
        sys.exit(1)

    if not args.confirm:
        print("\nDRY RUN complete - nothing was fit, evaluated, or written.")
        print("Pass --confirm to execute the 450-run grid against TEST (2015).")
        return

    print("\n--confirm passed - executing the 450-run grid against TEST (2015).\n")

    combos = [
        (model, regime, array, horizon, seed)
        for model in MODEL_RUNNERS
        for regime in REGIMES
        for array in ARRAYS
        for horizon in HORIZONS
        for seed in SEEDS
    ]
    total = len(combos)

    TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_run = 0
    n_skipped = 0
    n_failed = 0
    sweep_start = time.perf_counter()

    for i, (model, regime, array, horizon, seed) in enumerate(combos, start=1):
        config_probe = {
            "model": model,
            "array": array,
            "horizon": horizon,
            "regime": regime,
            "seed": seed,
            "eval_split": "test",
        }
        run_id = make_run_id(config_probe)
        out_path = TEST_RESULTS_DIR / f"{run_id}.json"
        prefix = f"[{i:3d}/{total}] {run_id}"

        if out_path.exists():
            print(f"{prefix}  skip (already exists)")
            n_skipped += 1
            continue

        run_start = time.perf_counter()
        try:
            config, metrics, timings, extra = MODEL_RUNNERS[model](array, horizon, regime, seed)
            write_run(config, metrics, timings, extra=extra, results_dir=str(TEST_RESULTS_DIR))
        except Exception as exc:
            print(f"{prefix}  FAILED: {exc}")
            n_failed += 1
            continue
        run_seconds = time.perf_counter() - run_start

        d = metrics["daylight"]
        print(
            f"{prefix}  {run_seconds:6.1f}s  "
            f"skill_vs_convex={d['skill_vs_convex']:+.4f}  "
            f"skill_vs_persistence={d['skill_vs_persistence']:+.4f}  "
            f"rmse={d['rmse']:.4f}  n_excluded_outage={timings['n_excluded_outage']}"
        )
        n_run += 1

    total_minutes = (time.perf_counter() - sweep_start) / 60
    print(
        f"\nfinal test run done: {n_run} run, {n_skipped} skipped (already existed), "
        f"{n_failed} failed, {total_minutes:.1f} min elapsed"
    )
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
