"""Training-length ablation for the residual-correction penalty: does
lstm_residual score below plain lstm because residual correction does not
help (claim A), or because the out-of-fold scheme (CLAUDE.md rule 6) is
weak with only 2 folds on the default 3-year TRAIN_YEARS - the fold
predicting 2013 saw only 2011-2012, a third less data than the deployed
base model that saw all of 2011-2013 (claim B)?

This reruns the same lstm / lstm_residual comparison with TRAIN_YEARS =
(2009, 2010, 2011, 2012, 2013): 5 years, 4 folds (2010, 2011, 2012, 2013
each held out in turn), so the weakest fold's base model saw 4 years
against the final model's 5 - proportionally much closer than 2 folds on
3 years. Arrays 11 and 12 only: array17 was installed March 2010 and has
no clean 2009 data (src/data/splits.py's TRAIN_YEARS comment).

This depends on two fixes made alongside this script (both needed for the
ablation to run at all / run correctly, see their own docstrings/comments):
  - src/data/clearsky_power.py fit_temperature_climatology no longer
    requires df_train's years to equal the default TRAIN_YEARS exactly,
    only that they exclude VAL_YEARS/TEST_YEARS.
  - src/models/residual.py ResidualCorrected._oof_residuals now derives
    fold years from df_train's own contents instead of re-filtering
    against the hardcoded TRAIN_YEARS constant, so a wider training window
    actually produces more folds instead of being silently truncated back
    to 2011-2013.

RUN ID COLLISION (make_run_id, src/eval/runner.py, does not encode the
training window): every run this script writes goes to
results/train5yr/<run_id>.json, a SEPARATE subdirectory from results/ - the
existing 225-run seed sweep under results/ is untouched. run_experiment
still records the effective train_years in config for anyone reading a
train5yr/*.json directly.

Grid: models [lstm, lstm_residual] x arrays [array11, array12] x
horizons [1, 3, 6] x seeds [0, 1, 2] = 36 runs. XGBoost is not part of this
grid - the comparison of interest is lstm_residual minus lstm WITHIN the
same training window, not a cross-model-family comparison.

After the sweep, prints per array x horizon:
  lstm_residual - lstm under TRAIN_YEARS 2011-2013 (existing runs in
    results/, restricted to seeds 0-2 for a like-for-like comparison with
    this script's 3-seed grid)
  lstm_residual - lstm under TRAIN_YEARS 2009-2013 (this script's new runs
    in results/train5yr/)
  the change between the two deltas
  n_oof_residuals and fold count for both settings (from lstm_residual's
    config; identical across seeds for a given array/horizon/train_years,
    so only seed 0's value is shown)

Usage:
    python scripts/rerun_residual_5yr.py
"""

import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.eval.runner import make_run_id  # noqa: E402

import run_lstm_dev  # noqa: E402
import run_residual_dev  # noqa: E402

RESULTS_DIR_3YR = REPO_ROOT / "results"
RESULTS_DIR_5YR = REPO_ROOT / "results" / "train5yr"

ARRAYS = ["array11", "array12"]
HORIZONS = [1, 3, 6]
SEEDS = [0, 1, 2]
REGIME = "lagged"

TRAIN_YEARS_3YR = (2011, 2012, 2013)
TRAIN_YEARS_5YR = (2009, 2010, 2011, 2012, 2013)

MODEL_RUNNERS = {
    "lstm": lambda array, horizon, seed, results_dir: run_lstm_dev.run_experiment(
        array, horizon, REGIME, seed,
        train_years=TRAIN_YEARS_5YR, results_dir=results_dir, verbose=False,
    ),
    "lstm_residual": lambda array, horizon, seed, results_dir: run_residual_dev.run_experiment(
        array, horizon, REGIME, seed, base="lstm", residual_fit_split="oof",
        train_years=TRAIN_YEARS_5YR, results_dir=results_dir, verbose=False,
    ),
}


def run_sweep():
    combos = [
        (model, array, horizon, seed)
        for model in MODEL_RUNNERS
        for array in ARRAYS
        for horizon in HORIZONS
        for seed in SEEDS
    ]
    total = len(combos)
    print(
        f"5-year ablation sweep: {total} runs planned "
        f"(models={list(MODEL_RUNNERS)}, arrays={ARRAYS}, horizons={HORIZONS}, "
        f"seeds={SEEDS}, train_years={TRAIN_YEARS_5YR})\n"
        f"writing to {RESULTS_DIR_5YR} (existing results/ untouched)\n"
    )

    n_run = 0
    n_skipped = 0
    n_failed = 0
    sweep_start = time.perf_counter()

    for i, (model, array, horizon, seed) in enumerate(combos, start=1):
        config = {
            "model": model,
            "array": array,
            "horizon": horizon,
            "regime": REGIME,
            "seed": seed,
            "eval_split": "val",
        }
        run_id = make_run_id(config)
        out_path = RESULTS_DIR_5YR / f"{run_id}.json"
        prefix = f"[{i:3d}/{total}] {run_id}"

        if out_path.exists():
            print(f"{prefix}  skip (already exists)")
            n_skipped += 1
            continue

        run_start = time.perf_counter()
        try:
            _, metrics = MODEL_RUNNERS[model](array, horizon, seed, RESULTS_DIR_5YR)
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
            f"rmse={d['rmse']:.4f}"
        )
        n_run += 1

    total_minutes = (time.perf_counter() - sweep_start) / 60
    print(
        f"\nsweep done: {n_run} run, {n_skipped} skipped (already existed), "
        f"{n_failed} failed, {total_minutes:.1f} min elapsed"
    )
    return n_failed


def load_skill(results_dir, model, array, horizon, seeds):
    values = []
    for seed in seeds:
        run_id = f"{model}_{array}_h{horizon}_{REGIME}_seed{seed}"
        path = Path(results_dir) / f"{run_id}.json"
        if not path.exists():
            continue
        with open(path) as f:
            record = json.load(f)
        values.append(record["metrics"]["daylight"]["skill_vs_convex"])
    return values


def load_oof_info(results_dir, array, horizon, seed=0):
    run_id = f"lstm_residual_{array}_h{horizon}_{REGIME}_seed{seed}"
    path = Path(results_dir) / f"{run_id}.json"
    if not path.exists():
        return None, None
    with open(path) as f:
        record = json.load(f)
    cfg = record["config"]
    n_oof = cfg.get("n_oof_residuals")
    oof_years = cfg.get("oof_years")
    n_folds = len(oof_years) if oof_years is not None else None
    return n_oof, n_folds


def print_comparison():
    print("\n" + "=" * 96)
    print("lstm_residual - lstm (mean skill_vs_convex, daylight, seeds 0-2), 3-year vs 5-year TRAIN_YEARS")
    print("=" * 96)

    for array in ARRAYS:
        for horizon in HORIZONS:
            lstm_3yr = load_skill(RESULTS_DIR_3YR, "lstm", array, horizon, SEEDS)
            resid_3yr = load_skill(RESULTS_DIR_3YR, "lstm_residual", array, horizon, SEEDS)
            lstm_5yr = load_skill(RESULTS_DIR_5YR, "lstm", array, horizon, SEEDS)
            resid_5yr = load_skill(RESULTS_DIR_5YR, "lstm_residual", array, horizon, SEEDS)

            print(f"\n{array}  h{horizon}")

            if len(lstm_3yr) < len(SEEDS) or len(resid_3yr) < len(SEEDS):
                print(f"  3yr : incomplete existing runs (lstm n={len(lstm_3yr)}, "
                      f"lstm_residual n={len(resid_3yr)}) - skipping")
                delta_3yr = None
            else:
                delta_3yr = statistics.mean(resid_3yr) - statistics.mean(lstm_3yr)
                print(f"  3yr : lstm={statistics.mean(lstm_3yr):+.4f}  "
                      f"lstm_residual={statistics.mean(resid_3yr):+.4f}  delta={delta_3yr:+.4f}")

            if len(lstm_5yr) < len(SEEDS) or len(resid_5yr) < len(SEEDS):
                print(f"  5yr : incomplete new runs (lstm n={len(lstm_5yr)}, "
                      f"lstm_residual n={len(resid_5yr)}) - skipping")
                delta_5yr = None
            else:
                delta_5yr = statistics.mean(resid_5yr) - statistics.mean(lstm_5yr)
                print(f"  5yr : lstm={statistics.mean(lstm_5yr):+.4f}  "
                      f"lstm_residual={statistics.mean(resid_5yr):+.4f}  delta={delta_5yr:+.4f}")

            if delta_3yr is not None and delta_5yr is not None:
                print(f"  change in delta (5yr - 3yr): {delta_5yr - delta_3yr:+.4f}")

            n_oof_3yr, n_folds_3yr = load_oof_info(RESULTS_DIR_3YR, array, horizon)
            n_oof_5yr, n_folds_5yr = load_oof_info(RESULTS_DIR_5YR, array, horizon)
            print(f"  n_oof_residuals: 3yr={n_oof_3yr} ({n_folds_3yr} folds)   "
                  f"5yr={n_oof_5yr} ({n_folds_5yr} folds)")


def main():
    n_failed = run_sweep()
    print_comparison()
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
