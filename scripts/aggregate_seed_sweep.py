"""Aggregate scripts/run_seed_sweep.py's results/<run_id>.json files into
a seed-variance summary: mean and std of skill_vs_convex,
skill_vs_persistence, and daylight RMSE, per model x array x horizon,
across the 5 sweep seeds.

Reads results/<model>_<array>_h<horizon>_lagged_seed<seed>.json for every
(model, array, horizon, seed) in the same grid run_seed_sweep.py sweeps.
A cell with fewer than 5 seeds present (sweep not finished, or a run
failed) is still reported, with n_seeds showing how many contributed.

Writes results/seed_sweep_summary.csv (one row per model x array x
horizon) and, for each array x horizon, prints the difference in mean
skill_vs_convex for every pairwise model comparison (not just LSTM vs
XGBoost - now that CNN-LSTM is a third model, all three pairs are
reported), flagging whether it exceeds 2 standard deviations of either
model's own seed-to-seed spread - the bar for treating the gap as a real
difference rather than seed noise.

Usage:
    python scripts/aggregate_seed_sweep.py
"""

import csv
import itertools
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

MODELS = ["xgboost", "lstm", "cnn_lstm"]
ARRAYS = ["array11", "array12", "array17"]
HORIZONS = [1, 3, 6]
SEEDS = [0, 1, 2, 3, 4]
REGIME = "lagged"

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


def load_daylight_metrics():
    """(model, array, horizon) -> list of metrics['daylight'] dicts, one
    per seed found on disk.
    """
    found = {}
    for model in MODELS:
        for array in ARRAYS:
            for horizon in HORIZONS:
                metrics_list = []
                for seed in SEEDS:
                    run_id = f"{model}_{array}_h{horizon}_{REGIME}_seed{seed}"
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
    with three models there are three pairs per array x horizon:
    xgboost-vs-lstm, xgboost-vs-cnn_lstm, lstm-vs-cnn_lstm.
    """
    print("\npairwise skill_vs_convex comparisons, per array x horizon:")
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
                a_2std = 2 * a["std_skill_vs_convex"]
                b_2std = 2 * b["std_skill_vs_convex"]
                exceeds_a = abs(diff) > a_2std
                exceeds_b = abs(diff) > b_2std

                if exceeds_a and exceeds_b:
                    verdict = "exceeds 2*std of BOTH -> likely a real difference"
                elif exceeds_a or exceeds_b:
                    verdict = "exceeds 2*std of ONE model only -> borderline"
                else:
                    verdict = "within 2*std of both -> not distinguishable from seed noise"

                print(
                    f"    {model_b} - {model_a} = {diff:+.4f}  "
                    f"({model_a} 2*std={a_2std:.4f}, {model_b} 2*std={b_2std:.4f})  {verdict}"
                )


def main():
    metrics_by_key = load_daylight_metrics()
    n_found = sum(len(v) for v in metrics_by_key.values())
    n_expected = len(MODELS) * len(ARRAYS) * len(HORIZONS) * len(SEEDS)
    print(f"found {n_found}/{n_expected} run JSONs for this grid\n")

    summary = summarize(metrics_by_key)
    print_summary_table(summary)

    out_path = RESULTS_DIR / "seed_sweep_summary.csv"
    write_csv(summary, out_path)
    print(f"\nwrote {out_path}")

    print_pairwise_comparisons(summary)


if __name__ == "__main__":
    main()
