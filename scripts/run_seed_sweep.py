"""Seed variance sweep for XGBoost, LSTM, and CNN-LSTM.

Motivation: a single-seed comparison (e.g. LSTM +0.279 vs XGBoost +0.276
skill_vs_convex, array11 h3 lagged) cannot distinguish a real difference
from noise. This script re-runs the same grid across 5 seeds so
scripts/aggregate_seed_sweep.py can report a mean and standard deviation
per model x array x horizon.

Grid: models [xgboost, lstm, cnn_lstm] x arrays [array11, array12,
array17] x horizons [1, 3, 6] x regime [lagged] x seeds [0, 1, 2, 3, 4] =
135 runs. Evaluated on the VALIDATION split (2014) only - this script
never reads 2015, exactly like run_xgb_dev.py, run_lstm_dev.py, and
run_cnn_lstm_dev.py, which it calls directly.

Rather than duplicate the data-loading/model-fit/metrics pipeline here,
this script imports run_experiment() from scripts/run_xgb_dev.py,
scripts/run_lstm_dev.py, and scripts/run_cnn_lstm_dev.py. Those three
functions were pulled out of each script's main() specifically so they
could be called in a loop like this without copy-pasting the pipeline a
third time. The setup that was IDENTICAL between run_xgb_dev.py and
run_lstm_dev.py (ARRAYS, load_and_prepare, add_clearsky_power_per_split)
was moved out further, into src/data/pipeline.py, which all three dev
scripts now import from too - see that module's docstring.

Each run writes its own results/<run_id>.json exactly as run_xgb_dev.py,
run_lstm_dev.py, and run_cnn_lstm_dev.py do on their own (same schema,
same write_run call). Runs whose JSON already exists are skipped without
re-computing anything - see write_run's own FileExistsError-on-overwrite
guard in src/eval/runner.py, which this script relies on by checking
existence first rather than catching that error.

Usage:
    python scripts/run_seed_sweep.py
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.eval.runner import make_run_id  # noqa: E402

import run_cnn_lstm_dev  # noqa: E402
import run_lstm_dev  # noqa: E402
import run_xgb_dev  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"

ARRAYS = ["array11", "array12", "array17"]
HORIZONS = [1, 3, 6]
SEEDS = [0, 1, 2, 3, 4]
REGIME = "lagged"

# model name (matches XGBForecaster.name / LSTMForecaster.name /
# CNNLSTMForecaster.name, and the run_id string) -> the
# run_experiment(array, horizon, regime, seed, verbose=...) function that
# produces it. run_cnn_lstm_dev.run_experiment takes extra n_filters/
# kernel_size arguments, but both have defaults, so calling it with the
# same (array, horizon, regime, seed, verbose=...) signature as the other
# two still works.
MODEL_RUNNERS = {
    "xgboost": run_xgb_dev.run_experiment,
    "lstm": run_lstm_dev.run_experiment,
    "cnn_lstm": run_cnn_lstm_dev.run_experiment,
}


def main():
    combos = [
        (model, array, horizon, seed)
        for model in MODEL_RUNNERS
        for array in ARRAYS
        for horizon in HORIZONS
        for seed in SEEDS
    ]
    total = len(combos)
    print(
        f"seed sweep: {total} runs planned "
        f"(models={list(MODEL_RUNNERS)}, arrays={ARRAYS}, horizons={HORIZONS}, "
        f"regime={REGIME!r}, seeds={SEEDS})\n"
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
        out_path = RESULTS_DIR / f"{run_id}.json"
        prefix = f"[{i:3d}/{total}] {run_id}"

        if out_path.exists():
            print(f"{prefix}  skip (already exists)")
            n_skipped += 1
            continue

        run_start = time.perf_counter()
        try:
            _, metrics = MODEL_RUNNERS[model](array, horizon, REGIME, seed, verbose=False)
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
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
