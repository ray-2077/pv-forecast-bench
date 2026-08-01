"""Build the paper's sky-condition table (RQ3): does forecast accuracy
depend on atmospheric condition, not just horizon and array?

Sky condition is classified per src.eval.sky.classify_sky - from k_ghi and
its short-window variability ONLY, never from k_p. See that module's
docstring for why k_p would be the wrong choice (array-specific, and
computed from the forecasting target itself). Classification uses target-
time k_ghi, which is legitimate ONLY as a post-hoc stratification of
already-computed forecast errors - it is never fed into any model.

Pipeline, per array, VALIDATION split (2014) only - this script never
reads 2015 (CLAUDE.md research-integrity rule, test set touched once at
the end):
  load one array's processed parquet -> solar position / clear-sky
  irradiance / daylight mask / clear-sky index of GHI (src.data.clearsky)
  -> chronological split (src.data.splits) -> temperature climatology and
  gain fit on TRAIN only, clear-sky POWER added to train and val
  (src.data.clearsky_power) -> classify every row of val by sky condition
  (once per array - sky condition does not depend on horizon) -> for each
  horizon, fit xgboost, lstm, lstm_residual (residual uses
  residual_fit_split='oof', the corrected scheme, CLAUDE.md rule 6) plus
  the convex reference AT SEED 0 -> restrict to the intersection of
  timestamps where all four produced a prediction, further restricted to
  is_daylight and with documented-outage hours dropped
  (src.eval.exclusions) -> within that evaluation set, group by sky class
  and compute RMSE, nRMSE, and skill vs. the convex reference per model.

Prints, per array, the count and percentage of each sky class over every
daylight validation hour (independent of horizon or model coverage), plus
the same totalled across all three arrays. Writes results/table_sky.csv,
one row per (array, horizon, sky_class, model): array, horizon, sky_class,
n, model, rmse, nrmse, skill_vs_convex.

No plotting, no test-split access, no fabricated numbers - every row is a
real model fit and predict on real data (CLAUDE.md research integrity
rules).

Usage:
    python scripts/build_table_sky.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.pipeline import ARRAYS, add_clearsky_power_per_split, load_and_prepare
from src.data.splits import split_chronological
from src.eval import metrics
from src.eval.exclusions import exclusion_mask
from src.eval.runner import set_all_seeds
from src.eval.sky import CATEGORIES, classify_sky, sky_class_counts
from src.models.base import check_no_lookahead
from src.models.climatology import ConvexCombination
from src.models.lstm import LSTMForecaster
from src.models.residual import ResidualCorrected
from src.models.xgb import XGBForecaster

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

HORIZONS = (1, 3, 6)
SEED = 0
REGIME = "lagged"
REPORTED_MODELS = ("xgboost", "lstm", "lstm_residual")

CSV_COLUMNS = [
    "array", "horizon", "sky_class", "n", "model", "rmse", "nrmse", "skill_vs_convex",
]


def fit_models(train, val, horizon, seed):
    """Fit xgboost, lstm, lstm_residual (residual_fit_split='oof', CLAUDE.md
    rule 6) plus the convex reference at `seed`, on this (array, horizon)'s
    train/val. Mirrors scripts/build_table6_dm.py's fit_all_models, minus
    cnn_lstm/cnn_lstm_residual/smart_persistence, which this table does not
    report.
    """
    set_all_seeds(seed)
    preds = {}

    xgb_model = XGBForecaster(seed=seed, regime=REGIME)
    xgb_model.fit(train, horizon, df_val=val)
    preds["xgboost"] = xgb_model.predict(val, horizon)
    check_no_lookahead(val, preds["xgboost"], horizon)

    lstm_model = LSTMForecaster(seed=seed, regime=REGIME)
    lstm_model.fit(train, horizon, df_val=val)
    preds["lstm"] = lstm_model.predict(val, horizon)
    check_no_lookahead(val, preds["lstm"], horizon)

    lstm_residual_model = ResidualCorrected(
        LSTMForecaster(seed=seed, regime=REGIME), seed=seed, residual_fit_split="oof"
    )
    lstm_residual_model.fit(train, horizon, val)
    preds["lstm_residual"] = lstm_residual_model.predict(val, horizon)
    check_no_lookahead(val, preds["lstm_residual"], horizon)

    convex_model = ConvexCombination()
    convex_model.fit(train, horizon, val)
    preds["convex_reference"] = convex_model.predict(val, horizon)
    check_no_lookahead(val, preds["convex_reference"], horizon)

    return preds


def build_cell(array, horizon, train, val, nameplate_kw, sky):
    """One (array, horizon) cell: fit the 3 reported models plus the
    convex reference, restrict to the shared, daylight, non-outage
    evaluation set, then compute RMSE/nRMSE/skill_vs_convex for each
    reported model within each sky class. Returns a list of row dicts.
    """
    preds = fit_models(train, val, horizon, SEED)

    common_idx = preds["xgboost"].index
    for name, p in preds.items():
        if name == "xgboost":
            continue
        common_idx = common_idx.intersection(p.index)

    daylight_mask = val.loc[common_idx, "is_daylight"].to_numpy()
    daylight_idx = common_idx[daylight_mask]

    outage_mask = exclusion_mask(array, daylight_idx)
    eval_idx = daylight_idx[~outage_mask.to_numpy()]

    y_true = val.loc[eval_idx, "Active_Power"]
    y_ref = preds["convex_reference"].loc[eval_idx]
    sky_eval = sky.loc[eval_idx]

    print(
        f"array={array}  horizon={horizon}  n_intersection(4 models)={len(common_idx)}  "
        f"n_daylight={len(daylight_idx)}  n_eval={len(eval_idx)}  "
        f"n_unclassified_sky={int(sky_eval.isna().sum())}"
    )

    rows = []
    for sky_class in CATEGORIES:
        class_idx = sky_eval[sky_eval == sky_class].index
        n = len(class_idx)
        for model_name in REPORTED_MODELS:
            if n == 0:
                rows.append({
                    "array": array, "horizon": horizon, "sky_class": sky_class,
                    "n": 0, "model": model_name,
                    "rmse": float("nan"), "nrmse": float("nan"), "skill_vs_convex": float("nan"),
                })
                continue

            y_true_c = y_true.loc[class_idx]
            y_pred_c = preds[model_name].loc[class_idx]
            y_ref_c = y_ref.loc[class_idx]

            rows.append({
                "array": array,
                "horizon": horizon,
                "sky_class": sky_class,
                "n": n,
                "model": model_name,
                "rmse": metrics.rmse(y_true_c, y_pred_c),
                "nrmse": metrics.nrmse(y_true_c, y_pred_c, nameplate_kw),
                "skill_vs_convex": metrics.skill_score(y_true_c, y_pred_c, y_ref_c),
            })
    return rows


def print_class_proportions(label, counts):
    total = int(counts.sum())
    print(f"  {label}  (n_classified_daylight={total})")
    for sky_class in CATEGORIES:
        n = int(counts[sky_class])
        pct = 100.0 * n / total if total else float("nan")
        print(f"    {sky_class:14s} n={n:5d}  ({pct:5.1f}%)")


def main():
    all_rows = []
    overall_counts = pd.Series(0, index=CATEGORIES, dtype=int)

    print("=" * 90)
    print("sky class proportions per array (validation split, 2014, every daylight hour)")
    print("=" * 90)

    per_array_sky = {}
    for array in sorted(ARRAYS):
        df, nameplate_kw, gamma_pdc = load_and_prepare(array, PROCESSED_DIR)
        train, val, test = split_chronological(df)
        train, val, gain_info = add_clearsky_power_per_split(train, val, test, nameplate_kw, gamma_pdc)

        sky = classify_sky(val)
        counts = sky_class_counts(sky)
        overall_counts = overall_counts.add(counts, fill_value=0).astype(int)
        print_class_proportions(array, counts)

        per_array_sky[array] = (train, val, nameplate_kw, sky)

    print_class_proportions("overall (all 3 arrays)", overall_counts)

    for array in sorted(ARRAYS):
        train, val, nameplate_kw, sky = per_array_sky[array]
        for horizon in HORIZONS:
            print(f"\n{'=' * 90}")
            print(f"array={array}  horizon={horizon}h  seed={SEED}  regime={REGIME}  (validation split, 2014)")
            print("=" * 90)

            start = time.perf_counter()
            rows = build_cell(array, horizon, train, val, nameplate_kw, sky)
            elapsed = time.perf_counter() - start
            print(f"  cell done in {elapsed:.1f}s")

            all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows)[CSV_COLUMNS]
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "table_sky.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  ({len(out_df)} rows)")

    print(f"\n{'=' * 90}")
    print("array11 results (RMSE / nRMSE / skill_vs_convex per sky class)")
    print("=" * 90)
    array11 = out_df[out_df["array"] == "array11"]
    print(array11.to_string(index=False))


if __name__ == "__main__":
    main()
