"""Aggregate scripts/run_seed_sweep.py's results/<run_id>.json files into
a seed-variance summary: mean and std of skill_vs_convex,
skill_vs_persistence, and daylight RMSE, per model x array x horizon,
across the 5 sweep seeds.

Reads results/<model>_<array>_h<horizon>_<regime>_seed<seed>.json for
every (model, array, horizon, seed) in the same grid run_seed_sweep.py
sweeps - now five models: xgboost, lstm, cnn_lstm, lstm_residual,
cnn_lstm_residual (src/models/residual.py). --regime selects lagged
(default) or oracle, mirroring run_seed_sweep.py's flag exactly - see
CLAUDE.md rule 5, lagged and oracle must never be mixed in one table. A
cell with fewer than 5 seeds present (sweep not finished, or a run
failed) is still reported, with n_seeds showing how many contributed.

CHANGE LOG (2026-08-07): this script used to have no --regime flag at
all - REGIME was a hardcoded "lagged" constant, and results/
seed_sweep_summary.csv had no regime in its name. Running this script
with `--regime oracle` silently ignored the flag (there was no argparse,
so nothing even looked at sys.argv) and re-aggregated the lagged files
under a name that gave no indication anything was wrong. The output
looked like an oracle summary but was byte-identical to the lagged one
to four decimal places. Fixed by adding a real, strict argparse --regime
flag and moving the output to results/seed_sweep_summary_<regime>.csv so
the two can never shadow each other again.

Per src/models/residual.py's docstring, lstm_residual and
cnn_lstm_residual's own validation-split metrics are optimistic (their
residual stage is fit on validation residuals, per CLAUDE.md rule 6) -
this script reports their seed variance the same way as the other three
models for comparability, but that caveat applies to every row and
pairwise comparison involving either residual model below.

Writes results/seed_sweep_summary_<regime>.csv (one row per model x
array x horizon) and, for each array x horizon, prints the difference in
mean skill_vs_convex for every pairwise model comparison (with five
models, that is all 10 pairs, not just one or three).

This script used to also print a verdict ("exceeds 2*std of BOTH ->
likely a real difference", and similar) based on whether a pairwise
difference in means exceeded 2 standard deviations of either model's
5-seed spread. That heuristic is retired: it is not a valid two-sample
test (no null distribution, no p-value), and it measures the wrong
uncertainty besides - seed spread is TRAINING STOCHASTICITY (how much a
model's own accuracy varies when only its random initialization/shuffling
changes), not sampling uncertainty about whether one model truly forecasts
better on this evaluation period. Evidence it matters: array11 h6, LSTM
vs XGBoost - the 2*std heuristic called this "likely a real difference",
but the actual significance test (Diebold-Mariano, which accounts for the
autocorrelation of overlapping-horizon forecast errors) gives p_raw =
0.010, p_holm = 0.073 - not significant after Holm correction across all
pairs in that cell. See results/table6_dm.csv (scripts/build_table6_dm.py)
for the real test. Seed mean/std stays in this script and in Table 3: it
is a legitimate reproducibility statistic, just not a significance test.

Usage:
    python scripts/aggregate_seed_sweep.py [--regime {lagged,oracle}]
"""

import argparse
import csv
import itertools
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

MODELS = ["xgboost", "lstm", "cnn_lstm", "lstm_residual", "cnn_lstm_residual"]
ARRAYS = ["array11", "array12", "array17"]
HORIZONS = [1, 3, 6]
SEEDS = [0, 1, 2, 3, 4]

CSV_FIELDS = [
    "model",
    "array",
    "horizon",
    "n_seeds",
    "mean_skill_vs_convex",
    "std_skill_vs_convex",
    "mean_skill_vs_persistence",
    "std_skill_vs_persistence",
    "mean_rmse_daylight",
    "std_rmse_daylight",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=("lagged", "oracle"), default="lagged")
    # parse_args() (not parse_known_args()) is what makes an unrecognised
    # flag a hard error instead of being silently dropped - this is the
    # exact failure mode that made `--regime oracle` a no-op before this
    # script had any argparse at all. Keep it this way.
    return parser.parse_args()


def load_daylight_metrics(regime):
    """(model, array, horizon) -> list of metrics['daylight'] dicts, one
    per seed found on disk.
    """
    found = {}
    for model in MODELS:
        for array in ARRAYS:
            for horizon in HORIZONS:
                metrics_list = []
                for seed in SEEDS:
                    run_id = f"{model}_{array}_h{horizon}_{regime}_seed{seed}"
                    path = RESULTS_DIR / f"{run_id}.json"
                    if not path.exists():
                        continue
                    with open(path) as f:
                        record = json.load(f)
                    metrics_list.append(record["metrics"]["daylight"])
                found[(model, array, horizon)] = metrics_list
    return found


def mean_std(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def summarize(metrics_by_key):
    """(model, array, horizon) -> summary dict, skipping empty cells."""
    summary = {}
    for key, metrics_list in metrics_by_key.items():
        if not metrics_list:
            continue
        mean_convex, std_convex = mean_std([m["skill_vs_convex"] for m in metrics_list])
        mean_pers, std_pers = mean_std([m["skill_vs_persistence"] for m in metrics_list])
        mean_rmse, std_rmse = mean_std([m["rmse"] for m in metrics_list])
        summary[key] = {
            "n_seeds": len(metrics_list),
            "mean_skill_vs_convex": mean_convex,
            "std_skill_vs_convex": std_convex,
            "mean_skill_vs_persistence": mean_pers,
            "std_skill_vs_persistence": std_pers,
            "mean_rmse_daylight": mean_rmse,
            "std_rmse_daylight": std_rmse,
        }
    return summary


def print_summary_table(summary):
    print("per model x array x horizon (daylight subset, 5 seeds unless noted):\n")
    for model in MODELS:
        for array in ARRAYS:
            for horizon in HORIZONS:
                s = summary.get((model, array, horizon))
                if s is None:
                    print(f"  {model:8s} {array:8s} h{horizon}  no runs found")
                    continue
                note = "" if s["n_seeds"] == len(SEEDS) else f"  (only {s['n_seeds']}/{len(SEEDS)} seeds)"
                print(
                    f"  {model:8s} {array:8s} h{horizon}  "
                    f"skill_vs_convex={s['mean_skill_vs_convex']:+.4f}+/-{s['std_skill_vs_convex']:.4f}  "
                    f"skill_vs_persistence={s['mean_skill_vs_persistence']:+.4f}+/-{s['std_skill_vs_persistence']:.4f}  "
                    f"rmse={s['mean_rmse_daylight']:.4f}+/-{s['std_rmse_daylight']:.4f}"
                    f"{note}"
                )


def write_csv(summary, out_path):
    rows = []
    for (model, array, horizon), s in sorted(summary.items()):
        rows.append({"model": model, "array": array, "horizon": horizon, **s})
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_pairwise_comparisons(summary):
    """Every pairwise model comparison (not just one model vs another) -
    with five models (MODELS above) there are C(5,2) = 10 pairs per
    array x horizon, e.g. xgboost-vs-lstm, lstm-vs-lstm_residual,
    cnn_lstm-vs-cnn_lstm_residual, and so on for every combination.

    Prints only the difference in mean skill_vs_convex for each pair - no
    significance verdict. See the module docstring for why.
    """
    print("\npairwise skill_vs_convex comparisons, per array x horizon:")
    print(
        "Significance is tested by Diebold-Mariano in "
        "results/table6_dm.csv, not by seed spread. Seed spread measures "
        "training stochasticity; DM measures whether one model forecasts "
        "better on this data."
    )
    for array in ARRAYS:
        for horizon in HORIZONS:
            print(f"  {array:8s} h{horizon}")
            for model_a, model_b in itertools.combinations(MODELS, 2):
                a = summary.get((model_a, array, horizon))
                b = summary.get((model_b, array, horizon))
                if a is None or b is None:
                    print(f"    {model_a} vs {model_b}: missing data, cannot compare")
                    continue

                diff = b["mean_skill_vs_convex"] - a["mean_skill_vs_convex"]
                print(f"    {model_b} - {model_a} = {diff:+.4f}")


def main():
    regime = parse_args().regime

    metrics_by_key = load_daylight_metrics(regime)
    n_found = sum(len(v) for v in metrics_by_key.values())
    n_expected = len(MODELS) * len(ARRAYS) * len(HORIZONS) * len(SEEDS)
    print(f"regime={regime!r}: found {n_found}/{n_expected} run JSONs for this grid\n")

    summary = summarize(metrics_by_key)
    print_summary_table(summary)

    out_path = RESULTS_DIR / f"seed_sweep_summary_{regime}.csv"
    write_csv(summary, out_path)
    print(f"\nwrote {out_path}")

    print_pairwise_comparisons(summary)


if __name__ == "__main__":
    main()
